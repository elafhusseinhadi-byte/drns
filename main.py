from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import math
import random

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
    x = latitude  (مثلاً 30.5 .. 30.9)
    y = longitude (مثلاً 47.4 .. 47.9)
    """
    __tablename__ = "uav_state"

    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(Integer, unique=True, index=True, nullable=False)
    city = Column(String, default="Baghdad")
    x = Column(Float, nullable=False)   # latitude
    y = Column(Float, nullable=False)   # longitude
    altitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class UAVHistory(Base):
    __tablename__ = "uav_history"

    id = Column(Integer, primary_key=True, index=True)
    uav_id = Column(Integer, index=True, nullable=False)
    city = Column(String, default="Baghdad")
    x = Column(Float, nullable=False)   # latitude
    y = Column(Float, nullable=False)   # longitude
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
# 🧱 Pydantic Models
# ======================================

class UAVIn(BaseModel):
    uav_id: int
    x: float      # latitude
    y: float      # longitude
    altitude: float
    city: Optional[str] = "Baghdad"


class Prediction(BaseModel):
    t_seconds: float
    x: float
    y: float
    movement_km: float


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
    status: str
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
# ⚙️ إعداد التطبيق + Route رئيسية
# ======================================

app = FastAPI(
    title="UAV Server – Collision + Hybrid Predictive AI + Server-Side Avoidance",
    version="4.1"
)


@app.get("/")
def root():
    return {
        "status": "server online",
        "project": "UAV Collision + Predictive AI",
        "message": "Welcome to the UAV Cloud Server"
    }


# Thresholds
THR_COLLISION_KM = 1.0
INNER_NEAR_KM = 1.5
THR_NEAR_KM = 3.0


# ======================================
# 🧮 الدوال المساعدة
# ======================================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    حساب المسافة بين نقطتين على سطح الأرض بالكيلومتر
    (lat, lon) بالدرجات
    """
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_velocities(db: Session, history_window: int = 3) -> Dict[int, Dict[str, float]]:
    """
    حساب السرعة التقريبية لكل UAV من التاريخ (UAVHistory)
    نعتمد على آخر history_window نقاط (مثل 3)
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

        dx = newest.x - oldest.x   # lat diff
        dy = newest.y - oldest.y   # lon diff
        result[uav_row.uav_id] = {"vx": dx / dt, "vy": dy / dt}

    return result


def compute_conflicts(uavs: List[UAVState]):
    """
    حساب أقل مسافة لكل UAV + حالة الخطر + قائمة الـ conflicts
    (x = lat, y = lon)
    """
    n = len(uavs)
    info: Dict[int, Dict[str, Any]] = {
        u.uav_id: {"min_dist": float("inf"), "status": "safe", "conflicts": set()}
        for u in uavs
    }

    for i in range(n):
        ui = uavs[i]
        for j in range(i + 1, n):
            uj = uavs[j]

            d = haversine_km(ui.x, ui.y, uj.x, uj.y)

            if d < info[ui.uav_id]["min_dist"]:
                info[ui.uav_id]["min_dist"] = d
            if d < info[uj.uav_id]["min_dist"]:
                info[uj.uav_id]["min_dist"] = d

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

    for u in uavs:
        if info[u.uav_id]["min_dist"] == float("inf"):
            info[u.uav_id]["min_dist"] = 9999.0

    return info


def compute_server_avoidance(uavs: List[UAVState], conflict_info: Dict[int, Dict[str, Any]]):
    """
    خوارزمية التجنب (Server-Side Avoidance B+C) – نسخة مقوّاة
    تعتمد على:
    - متجه تنافر من الجيران (repulsion)
    - crowd factor حسب عدد الجيران
    - jitter بسيط حتى يمنع التماثل
    - scaling قوي حسب مستوى الخطر
    - حد أعلى للحركة (clamp)
    """
    id_to_uav = {u.uav_id: u for u in uavs}
    result: Dict[int, Optional[Avoidance]] = {}

    for u in uavs:
        info = conflict_info[u.uav_id]
        dmin = info["min_dist"]
        neighbors = list(info["conflicts"])

        if dmin >= THR_NEAR_KM or not neighbors:
            result[u.uav_id] = None
            continue

        ax = ay = 0.0
        for nid in neighbors:
            other = id_to_uav[nid]
            dx = u.x - other.x   # lat diff
            dy = u.y - other.y   # lon diff
            dist = math.hypot(dx, dy) + 1e-6
            ax += dx / dist
            ay += dy / dist

        # متوسط اتجاهات الجيران
        ax /= len(neighbors)
        ay /= len(neighbors)

        # 1) crowd-aware repulsion (إذا عنده جيران أكثر → دفع أقوى)
        crowd_factor = min(len(neighbors), 5)
        ax *= crowd_factor
        ay *= crowd_factor

        # 2) jitter خفيف حتى نكسر التماثل
        jitter = (random.random() - 0.5) * 0.0008
        ax += jitter
        ay += jitter

        # 3) scaling حسب مستوى الخطر
        if dmin < THR_COLLISION_KM:
            scale = 0.045
            note = "Ultra strong avoidance (collision zone)"
        elif dmin < INNER_NEAR_KM:
            scale = 0.025
            note = "Strong avoidance (inner near zone)"
        else:
            scale = 0.010
            note = "Moderate avoidance (outer near zone)"

        ax *= scale
        ay *= scale

        # 4) max movement clamp (حتى ما تطفر برّه المنطقة)
        max_step = 0.02  # تقريباً ~2 km
        step = math.hypot(ax, ay)
        if step > max_step:
            f = max_step / step
            ax *= f
            ay *= f

        result[u.uav_id] = Avoidance(
            suggested_dx=ax,
            suggested_dy=ay,
            note=note
        )

    return result


def compute_hybrid_predictions(
    uavs: List[UAVState],
    conflict_info: Dict[int, Dict[str, Any]],
    velocities: Dict[int, Dict[str, float]],
    t: float = 5.0
) -> Dict[int, Prediction]:
    """
    Hybrid Predictive AI (محسّن):
    - يعتمد على السرعة المحسوبة من الـ history
    - + متجه تنافر من الجيران (repulsion)
    - + scaling حسب مستوى الخطر (collision / near / safe)
    - + jitter صغير
    - يرجع Prediction لكل UAV مع movement_km
    """
    id_to_uav = {u.uav_id: u for u in uavs}
    preds: Dict[int, Prediction] = {}

    for u in uavs:
        info = conflict_info[u.uav_id]
        dmin = info["min_dist"]
        status = info["status"]
        neighbors = list(info["conflicts"])

        # ---- 1) Base velocity from history ----
        vel = velocities.get(u.uav_id, {"vx": 0.0, "vy": 0.0})
        vx = vel["vx"]   # lat speed
        vy = vel["vy"]   # lon speed

        # ---- 2) Repulsion from neighbors ----
        rx = ry = 0.0
        if neighbors:
            for nid in neighbors:
                other = id_to_uav[nid]
                dx = u.x - other.x
                dy = u.y - other.y
                dist = math.hypot(dx, dy) + 1e-6
                rx += dx / dist
                ry += dy / dist

            rx /= len(neighbors)
            ry /= len(neighbors)

        # ---- 3) Scaling حسب مستوى الخطر ----
        if status == "collision":
            w_vel = 0.6
            w_rep = 1.6
        elif status == "near":
            if dmin < INNER_NEAR_KM:
                w_vel = 0.9
                w_rep = 1.2
            else:  # outer near
                w_vel = 1.0
                w_rep = 0.8
        else:  # safe
            w_vel = 1.1
            w_rep = 0.3

        # ---- 4) Limit base velocity magnitude ----
        max_speed_deg = 0.0025  # حد أعلى تقريباً للسرعة بالـ degrees/second
        base_speed = math.hypot(vx, vy)
        if base_speed > 1e-9 and base_speed > max_speed_deg:
            scale = max_speed_deg / base_speed
            vx *= scale
            vy *= scale

        # ---- 5) Combine velocity + repulsion ----
        fx = w_vel * vx + w_rep * rx * 0.0015
        fy = w_vel * vy + w_rep * ry * 0.0015

        # ---- 6) Jitter صغير حتى ما يصير locking ----
        jitter_scale = 5e-5
        fx += random.uniform(-jitter_scale, jitter_scale)
        fy += random.uniform(-jitter_scale, jitter_scale)

        # ---- 7) Final predicted position ----
        x_new = u.x + fx * t   # lat new
        y_new = u.y + fy * t   # lon new

        movement_km = haversine_km(u.x, u.y, x_new, y_new)

        preds[u.uav_id] = Prediction(
            t_seconds=t,
            x=x_new,
            y=y_new,
            movement_km=round(movement_km, 5)
        )

    return preds


# ======================================
# PUT /uav
# ======================================

@app.put("/uav")
def put_uav(uav: UAVIn, db: Session = Depends(get_db)):
    """
    استقبال موقع UAV واحد وتحديث حالته + حفظها في الـ history
    """
    now = datetime.now(timezone.utc)

    state = db.query(UAVState).filter(UAVState.uav_id == uav.uav_id).first()

    if state is None:
        state = UAVState(
            uav_id=uav.uav_id,
            city=uav.city,
            x=uav.x,
            y=uav.y,
            altitude=uav.altitude,
            timestamp=now
        )
        db.add(state)
    else:
        state.city = uav.city
        state.x = uav.x
        state.y = uav.y
        state.altitude = uav.altitude
        state.timestamp = now

    hist = UAVHistory(
        uav_id=uav.uav_id,
        city=uav.city,
        x=uav.x,
        y=uav.y,
        altitude=uav.altitude,
        timestamp=now
    )
    db.add(hist)

    db.commit()
    db.refresh(state)

    return {"status": "ok", "uav_id": state.uav_id}


# ======================================
# GET /uavs
# ======================================

@app.get("/uavs", response_model=UAVListOut)
def get_uavs(process: bool = False, db: Session = Depends(get_db)):
    """
    إرجاع جميع الطائرات مع:
    - حالة الخطر (collision / near / safe)
    - أقل مسافة d_min
    - التنبؤ بالموقع (Hybrid Predictive AI)
    - معلومات التجنب إذا process=true
    """
    uavs = db.query(UAVState).order_by(UAVState.uav_id).all()

    if not uavs:
        return UAVListOut(count=0, uavs=[], collisions=0, near=0, safe=0)

    # 1) حساب التصادمات قبل/بعد التجنب
    conflict_info = compute_conflicts(uavs)

    # 2) تطبيق تجنب فعلي على المواقع إذا process = true
    if process:
        avoidance = compute_server_avoidance(uavs, conflict_info)
        now = datetime.now(timezone.utc)

        for u in uavs:
            avoid = avoidance.get(u.uav_id)
            if avoid:
                u.x += avoid.suggested_dx
                u.y += avoid.suggested_dy
                u.timestamp = now
                db.add(
                    UAVHistory(
                        uav_id=u.uav_id,
                        city=u.city,
                        x=u.x,
                        y=u.y,
                        altitude=u.altitude,
                        timestamp=now
                    )
                )

        db.commit()
        # إعادة تحميل الحالات بعد التحديث
        uavs = db.query(UAVState).order_by(UAVState.uav_id).all()
        conflict_info = compute_conflicts(uavs)
    else:
        avoidance = {u.uav_id: None for u in uavs}

    # 3) السرعات من الـ history
    velocities = compute_velocities(db)

    # 4) Hybrid Predictive AI داخل السيرفر
    hybrid_preds = compute_hybrid_predictions(uavs, conflict_info, velocities)

    out_list: List[UAVOut] = []
    counts = {"collision": 0, "near": 0, "safe": 0}

    for u in uavs:
        info = conflict_info[u.uav_id]
        status = info["status"]
        counts[status] += 1

        pred = hybrid_preds[u.uav_id]

        out_list.append(
            UAVOut(
                uav_id=u.uav_id,
                city=u.city,
                x=u.x,
                y=u.y,
                altitude=u.altitude,
                timestamp=u.timestamp,
                status=status,
                min_distance_km=round(info["min_dist"], 3),
                predicted=pred,
                avoidance=avoidance.get(u.uav_id),
                conflicts_with=list(info["conflicts"])
            )
        )

    return UAVListOut(
        count=len(out_list),
        uavs=out_list,
        collisions=counts["collision"],
        near=counts["near"],
        safe=counts["safe"]
    )


# ======================================
# GET /logs
# ======================================

@app.get("/logs")
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    """
    إرجاع آخر الحركات من جدول التاريخ UAVHistory
    """
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
# Health Check
# ======================================

@app.get("/health")
def health():
    return {"status": "ok"}
