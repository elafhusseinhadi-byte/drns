from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import math

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ======================================
# 🔗 قاعدة البيانات
# ======================================
DATABASE_URL = "sqlite:///./uav.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UAVState(Base):
    """
    حالة كل UAV الحالية (آخر موقع)
    """
    __tablename__ = "uav_state"

    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(Integer, unique=True, index=True, nullable=False)
    city = Column(String, default="Baghdad")
    x = Column(Float, nullable=False)        # longitude
    y = Column(Float, nullable=False)        # latitude
    altitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class UAVHistory(Base):
    """
    لوج كامل للحركات (لكل تحديث)
    """
    __tablename__ = "uav_history"

    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(Integer, index=True, nullable=False)
    city = Column(String, default="Baghdad")
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ======================================
# 🧱 Pydantic Models (IN / OUT)
# ======================================

class UAVIn(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    city: Optional[str] = "Baghdad"


class Prediction(BaseModel):
    t_seconds: float
    x: float
    y: float


class Avoidance(BaseModel):
    suggested_dx: float
    suggested_dy: float
    note: str


class UAVOut(BaseModel):
    uav_id: int
    city: str
    x: float
    y: float
    altitude: float
    timestamp: datetime
    status: str            # "safe" / "near" / "collision"
    min_distance_km: float
    predicted: Optional[Prediction] = None
    avoidance: Optional[Avoidance] = None
    conflicts_with: List[int] = []


class UAVListOut(BaseModel):
    count: int
    uavs: List[UAVOut]
    collisions: int
    near: int
    safe: int


# ======================================
# ⚙️ إعداد التطبيق
# ======================================

app = FastAPI(
    title="UAV Server – Collision + AI + Logging (Server-Side Avoidance)",
    version="3.0"
)

# Thresholds بالكيلومتر
THR_COLLISION_KM = 1.0       # تصادم
THR_NEAR_KM = 3.0            # نهاية منطقة الخطر
INNER_NEAR_KM = 1.5          # near الداخلي (اللي قلتي عليه)


# ======================================
# 🧮 دوال مساعدة: مسافة / سرعة / Conflicts / Avoidance
# ======================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    حساب مسافة حقيقية تقريباً بالكيلومتر بين نقطتين (lat, lon)
    """
    R = 6371.0  # نصف قطر الأرض بالكيلومتر
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_velocities(db: Session, history_window: int = 3) -> Dict[int, Dict[str, float]]:
    """
    "Predictive AI" بسيطة: تحسب السرعة المتوسطة من آخر N نقاط لكل UAV
    vx, vy = delta_pos / delta_time
    """
    result: Dict[int, Dict[str, float]] = {}

    for uav_row in db.query(UAVState).all():
        hist = (
            db.query(UAVHistory)
            .filter(UAVHistory.uav_id == uav_row.uav_id)
            .order_by(UAVHistory.timestamp.desc())
            .limit(history_window)
            .all()
        )

        if len(hist) < 2:
            result[uav_row.uav_id] = {"vx": 0.0, "vy": 0.0}
            continue

        newest = hist[0]
        oldest = hist[-1]
        dt = (newest.timestamp - oldest.timestamp).total_seconds()
        if dt <= 0:
            result[uav_row.uav_id] = {"vx": 0.0, "vy": 0.0}
            continue

        dx = newest.x - oldest.x
        dy = newest.y - oldest.y
        vx = dx / dt
        vy = dy / dt
        result[uav_row.uav_id] = {"vx": vx, "vy": vy}

    return result


def predict_position(uav: UAVState, vel: Dict[str, float], t: float = 5.0) -> Prediction:
    """
    توقع الموقع بعد t ثانية (إحداثيات، مو كم)
    """
    x_pred = uav.x + vel.get("vx", 0.0) * t
    y_pred = uav.y + vel.get("vy", 0.0) * t
    return Prediction(t_seconds=t, x=x_pred, y=y_pred)


def compute_conflicts(uavs: List[UAVState]) -> Dict[int, Dict[str, Any]]:
    """
    Multi-UAV conflict control:
    - حساب أقل مسافة لكل طائرة
    - تعيين حالة safe / near / collision
    - بناء قائمة من يكون قريب من منو
    """
    n = len(uavs)
    info: Dict[int, Dict[str, Any]] = {}

    for u in uavs:
        info[u.uav_id] = {
            "min_dist": float("inf"),
            "status": "safe",
            "conflicts": set(),  # set of uav_ids
        }

    for i in range(n):
        ui = uavs[i]
        for j in range(i + 1, n):
            uj = uavs[j]

            # مسافة بالكيلومتر
            d = haversine_km(ui.y, ui.x, uj.y, uj.x)

            # أقل مسافة
            if d < info[ui.uav_id]["min_dist"]:
                info[ui.uav_id]["min_dist"] = d
            if d < info[uj.uav_id]["min_dist"]:
                info[uj.uav_id]["min_dist"] = d

            # Near / Collision
            if d < THR_NEAR_KM:
                info[ui.uav_id]["conflicts"].add(uj.uav_id)
                info[uj.uav_id]["conflicts"].add(ui.uav_id)

            if d < THR_COLLISION_KM:
                info[ui.uav_id]["status"] = "collision"
                info[uj.uav_id]["status"] = "collision"
            elif d < THR_NEAR_KM:
                if info[ui.uav_id]["status"] != "collision":
                    info[ui.uav_id]["status"] = "near"
                if info[uj.uav_id]["status"] != "collision":
                    info[uj.uav_id]["status"] = "near"

    # إذا UAV وحدها، نخلي min_dist كبير
    for u in uavs:
        if info[u.uav_id]["min_dist"] == float("inf"):
            info[u.uav_id]["min_dist"] = 9999.0

    return info


def compute_server_avoidance(
    uavs: List[UAVState],
    conflict_info: Dict[int, Dict[str, Any]]
) -> Dict[int, Optional[Avoidance]]:
    """
    تجنّب داخل السيرفر:
    - يستخدم 3 مستويات:
      * d < 1 km        → Collision  → تجنب قوي
      * 1 ≤ d < 1.5 km  → Inner Near → تجنب متوسط (يغير اتجاهه بوضوح)
      * 1.5 ≤ d < 3 km  → Outer Near → تجنب خفيف
    - يرجع Avoidance لكل UAV (أو None)
    """
    id_to_uav = {u.uav_id: u for u in uavs}
    result: Dict[int, Optional[Avoidance]] = {}

    for u in uavs:
        info = conflict_info[u.uav_id]
        dmin = info["min_dist"]
        neighbors_ids = list(info["conflicts"])

        if dmin >= THR_NEAR_KM or not neighbors_ids:
            result[u.uav_id] = None
            continue

        # متّجه تنافري من الجيران
        ax = 0.0
        ay = 0.0
        for nid in neighbors_ids:
            other = id_to_uav[nid]
            dx = u.x - other.x
            dy = u.y - other.y
            dist = math.hypot(dx, dy) + 1e-6
            ax += dx / dist
            ay += dy / dist

        ax /= len(neighbors_ids)
        ay /= len(neighbors_ids)

        # اختيار قوة الدفع حسب المسافة (3-Zone Near)
        if dmin < THR_COLLISION_KM:
            # 🔴 Collision – قوي
            scale = 0.015
            note = "Strong avoidance (collision zone)"
        elif dmin < INNER_NEAR_KM:
            # 🟠 Near الداخلي
            scale = 0.008
            note = "Medium avoidance (inner near zone)"
        else:
            # 🟡 Near الخارجي
            scale = 0.003
            note = "Soft avoidance (outer near zone)"

        dx_apply = ax * scale
        dy_apply = ay * scale

        result[u.uav_id] = Avoidance(
            suggested_dx=dx_apply,
            suggested_dy=dy_apply,
            note=note
        )

    return result


# ======================================
# 🛰 PUT /uav  — إرسال أو تحديث طائرة (مع Logging)
# ======================================

@app.put("/uav", summary="Update or create UAV position (logging enabled)")
def put_uav(uav: UAVIn, db: Session = Depends(get_db)):
    """
    نفس اللي تستعمله من MATLAB:
    - يخزن آخر حالة في جدول uav_state
    - يسجل حركة جديدة في جدول uav_history (Logging)
    """
    now = datetime.now(timezone.utc)

    state = db.query(UAVState).filter(UAVState.uav_id == uav.uav_id).first()

    if state is None:
        state = UAVState(
            uav_id=uav.uav_id,
            city=uav.city or "Baghdad",
            x=uav.x,
            y=uav.y,
            altitude=uav.altitude,
            timestamp=now,
        )
        db.add(state)
    else:
        state.city = uav.city or state.city
        state.x = uav.x
        state.y = uav.y
        state.altitude = uav.altitude
        state.timestamp = now

    # Logging
    hist = UAVHistory(
        uav_id=uav.uav_id,
        city=uav.city or "Baghdad",
        x=uav.x,
        y=uav.y,
        altitude=uav.altitude,
        timestamp=now,
    )
    db.add(hist)

    db.commit()
    db.refresh(state)

    return {
        "status": "ok",
        "uav_id": state.uav_id,
        "timestamp": state.timestamp,
    }


# ======================================
# 📥 GET /uavs — مع خيار process للتجنب داخل السيرفر
# ======================================

@app.get(
    "/uavs",
    response_model=UAVListOut,
    summary="Get all UAVs with status, prediction and (optional) server-side avoidance"
)
def get_uavs(
    process: bool = False,  # إذا true: يطبق التجنب داخل السيرفر
    db: Session = Depends(get_db)
):
    # نجيب الحالة الحالية
    uavs = db.query(UAVState).order_by(UAVState.uav_id).all()

    if not uavs:
        return UAVListOut(count=0, uavs=[], collisions=0, near=0, safe=0)

    # --------- خطوة 1: نحسب الـ conflicts على الحالة الحالية ---------
    conflict_info = compute_conflicts(uavs)
    avoidance_dict: Dict[int, Optional[Avoidance]] = {}

    # --------- خطوة 2: إذا process=True نطبق التجنب ونحدّث الـ DB ---------
    if process:
        now = datetime.now(timezone.utc)

        # نحسب متّجهات التجنب حسب 3-Zone Near
        avoidance_dict = compute_server_avoidance(uavs, conflict_info)

        # نطبق الحركة على كل UAV ونضيف للـ history
        for u in uavs:
            avoid = avoidance_dict.get(u.uav_id)
            if avoid is None:
                continue

            u.x += avoid.suggested_dx
            u.y += avoid.suggested_dy
            u.timestamp = now

            hist = UAVHistory(
                uav_id=u.uav_id,
                city=u.city,
                x=u.x,
                y=u.y,
                altitude=u.altitude,
                timestamp=now,
            )
            db.add(hist)

        db.commit()

        # نعيد القراءة بعد الحركة
        uavs = db.query(UAVState).order_by(UAVState.uav_id).all()
        conflict_info = compute_conflicts(uavs)
    else:
        # إذا مكو تجنب، نخلي avoidance_dict فارغ (suggestion optional إذا حبيتي)
        avoidance_dict = {u.uav_id: None for u in uavs}

    # --------- خطوة 3: نحسب السرعات + التنبؤ ---------
    velocities = compute_velocities(db)

    out_list: List[UAVOut] = []
    counts = {"collision": 0, "near": 0, "safe": 0}

    for u in uavs:
        info = conflict_info[u.uav_id]
        status = info["status"]
        if status not in counts:
            counts[status] = 0
        counts[status] += 1

        vel = velocities.get(u.uav_id, {"vx": 0.0, "vy": 0.0})
        pred = predict_position(u, vel, t=5.0)

        # conflicts_with: نحول إلى list
        conflict_ids = list(info["conflicts"])
        avoid = avoidance_dict.get(u.uav_id)

        out = UAVOut(
            uav_id=u.uav_id,
            city=u.city,
            x=u.x,
            y=u.y,
            altitude=u.altitude,
            timestamp=u.timestamp,
            status=status,
            min_distance_km=round(info["min_dist"], 3),
            predicted=pred,
            avoidance=avoid,
            conflicts_with=conflict_ids,
        )
        out_list.append(out)

    return UAVListOut(
        count=len(out_list),
        uavs=out_list,
        collisions=counts.get("collision", 0),
        near=counts.get("near", 0),
        safe=counts.get("safe", 0),
    )


# ======================================
# 📜 GET /logs — آخر N حركة (Logging)
# ======================================

@app.get("/logs", summary="Get last N UAV logs")
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000")

    logs = (
        db.query(UAVHistory)
        .order_by(UAVHistory.timestamp.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": row.id,
            "uav_id": row.uav_id,
            "city": row.city,
            "x": row.x,
            "y": row.y,
            "altitude": row.altitude,
            "timestamp": row.timestamp,
        }
        for row in logs
    ]


# ======================================
# ❤️ Health Check بسيط
# ======================================

@app.get("/health", summary="Simple health check")
def health():
    return {"status": "ok"}
