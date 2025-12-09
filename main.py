from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone
import math, random

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ======================================
# DATABASE
# ======================================
DATABASE_URL = "sqlite:///./uav.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UAVState(Base):
    __tablename__ = "uav_state"
    id = Column(Integer, primary_key=True, index=True)

    uav_id = Column(Integer, unique=True, index=True, nullable=False)
    city = Column(String, default="Baghdad")

    # Position
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)

    # Ultra Pro: Kinematic state
    velocity = Column(Float, default=0.0)     # scalar speed
    heading = Column(Float, default=0.0)      # radians (0 = +x)

    ax = Column(Float, default=0.0)           # acceleration in x (same units as x)
    ay = Column(Float, default=0.0)           # acceleration in y

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class UAVHistory(Base):
    __tablename__ = "uav_history"
    id = Column(Integer, primary_key=True, index=True)

    uav_id = Column(Integer, index=True, nullable=False)
    city = Column(String, default="Baghdad")

    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    altitude = Column(Float, nullable=False)

    velocity = Column(Float, default=0.0)
    heading = Column(Float, default=0.0)

    ax = Column(Float, default=0.0)
    ay = Column(Float, default=0.0)

    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ======================================
# MODELS
# ======================================
class UAVIn(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    city: Optional[str] = "Baghdad"

    # Ultra Pro inputs من المحاكاة
    velocity: Optional[float] = 0.0
    heading: Optional[float] = 0.0   # يفضّل بالراديان (rad)
    ax: Optional[float] = 0.0
    ay: Optional[float] = 0.0


class Prediction(BaseModel):
    t_seconds: float
    x: float
    y: float


class Avoidance(BaseModel):
    suggested_dx: float
    suggested_dy: float
    suggested_dv: float = 0.0        # تغيير السرعة المقترح
    suggested_dheading: float = 0.0  # تغيير الزاوية المقترح (rad)
    note: str


class UAVOut(BaseModel):
    uav_id: int
    city: str
    x: float
    y: float
    altitude: float
    timestamp: datetime

    # الحالة الحركية الحالية
    velocity: float
    heading: float
    ax: float
    ay: float

    status: str
    min_distance_km: float

    # تنبؤ قصير + مسار مستقبلي
    predicted: Optional[Prediction] = None
    future_path: List[Prediction] = []

    avoidance: Optional[Avoidance] = None
    conflicts_with: List[int] = []


class UAVListOut(BaseModel):
    count: int
    uavs: List[UAVOut]
    collisions: int
    near: int
    safe: int


app = FastAPI()

# ======================================
# ROOT
# ======================================
@app.get("/")
def root():
    return {"status": "server online", "project": "UAV Cloud System – Ultra Pro"}


# ======================================
# RESET FULL DATABASE
# ======================================
@app.delete("/reset")
def reset_uavs(db: Session = Depends(get_db)):
    db.query(UAVState).delete()
    db.query(UAVHistory).delete()
    db.commit()
    return {"status": "database_cleared"}


# ======================================
# INIT — Sequential UAVs
# ======================================
@app.post("/init")
def init_uavs(n: int, db: Session = Depends(get_db)):
    db.query(UAVState).delete()
    db.query(UAVHistory).delete()
    db.commit()

    for i in range(1, n + 1):
        u = UAVState(
            uav_id=i,
            city="Baghdad",
            x=30.80,
            y=47.50,
            altitude=200.0,
            velocity=0.0,
            heading=0.0,
            ax=0.0,
            ay=0.0,
            timestamp=datetime.utcnow()
        )
        db.add(u)

    db.commit()
    return {"status": "initialized", "count": n}


# ======================================
# INIT RANDOM — UAV Random Spread
# ======================================
@app.post("/init_random")
def init_uavs_random(n: int, db: Session = Depends(get_db)):
    if n <= 0:
        raise HTTPException(status_code=400, detail="n must be > 0")

    db.query(UAVState).delete()
    db.query(UAVHistory).delete()
    db.commit()

    # Baghdad bounding box
    min_x, max_x = 30.55, 30.95
    min_y, max_y = 47.25, 47.75

    for i in range(1, n + 1):
        u = UAVState(
            uav_id=i,
            city="Baghdad",
            x=random.uniform(min_x, max_x),
            y=random.uniform(min_y, max_y),
            altitude=random.uniform(180, 230),
            velocity=random.uniform(0.0005, 0.0020),  # مثال بسيط
            heading=random.uniform(0.0, 2 * math.pi),
            ax=0.0,
            ay=0.0,
            timestamp=datetime.utcnow()
        )
        db.add(u)

    db.commit()
    return {"status": "random_initialized", "count": n}


# ======================================
# DISTANCE CONSTANTS
# ======================================
THR_COLLISION_KM = 1.0
INNER_NEAR_KM = 1.5
THR_NEAR_KM = 3.0


# ======================================
# HAVERSINE
# ======================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# ======================================
# KINEMATIC PREDICTION
# x,y + v, heading + (ax, ay)
# ======================================
def predict_state(uav: UAVState, t: float) -> Prediction:
    """
    تنبؤ شديد البساطة:
    - velocity: سرعة قياسية
    - heading: زاوية باتجاه الحركة (rad)
    - ax, ay: تعجيل في محور x,y
    """

    v = uav.velocity or 0.0
    theta = uav.heading or 0.0

    ax = uav.ax or 0.0
    ay = uav.ay or 0.0

    # نفكك السرعة إلى مكونات أولية
    vx0 = v * math.cos(theta)
    vy0 = v * math.sin(theta)

    # قانون الحركة: x = x0 + v0*t + 0.5*a*t^2
    x_future = uav.x + vx0 * t + 0.5 * ax * t * t
    y_future = uav.y + vy0 * t + 0.5 * ay * t * t

    return Prediction(
        t_seconds=t,
        x=x_future,
        y=y_future
    )


def predict_future_path(uav: UAVState, horizons: List[float]) -> List[Prediction]:
    return [predict_state(uav, t) for t in horizons]


# ======================================
# CONFLICT DETECTION (حالي فقط)
# ======================================
def compute_conflicts(uavs: List[UAVState]):
    info = {
        u.uav_id: {"min_dist": float("inf"), "status": "safe", "conflicts": set()}
        for u in uavs
    }

    for i in range(len(uavs)):
        ui = uavs[i]
        for j in range(i + 1, len(uavs)):
            uj = uavs[j]

            # نعتبر x=lon, y=lat مثل السابق
            d = haversine_km(ui.y, ui.x, uj.y, uj.x)

            info[ui.uav_id]["min_dist"] = min(info[ui.uav_id]["min_dist"], d)
            info[uj.uav_id]["min_dist"] = min(info[uj.uav_id]["min_dist"], d)

            if d < THR_NEAR_KM:
                info[ui.uav_id]["conflicts"].add(uj.uav_id)
                info[uj.uav_id]["conflicts"].add(ui.uav_id)

            if d < THR_COLLISION_KM:
                info[ui.uav_id]["status"] = "collision"
                info[uj.uav_id]["status"] = "collision"
            elif d < INNER_NEAR_KM:
                if info[ui.uav_id]["status"] != "collision":
                    info[ui.uav_id]["status"] = "inner_near"
                if info[uj.uav_id]["status"] != "collision":
                    info[uj.uav_id]["status"] = "inner_near"
            elif d < THR_NEAR_KM:
                if info[ui.uav_id]["status"] not in ("collision", "inner_near"):
                    info[ui.uav_id]["status"] = "outer_near"
                if info[uj.uav_id]["status"] not in ("collision", "inner_near"):
                    info[uj.uav_id]["status"] = "outer_near"

    for u in uavs:
        if info[u.uav_id]["min_dist"] == float("inf"):
            info[u.uav_id]["min_dist"] = 9999.0

    return info


# ======================================
# SERVER AVOIDANCE – Velocity Aware
# ======================================
def compute_server_avoidance(uavs: List[UAVState], conflict_info):
    id_to_uav = {u.uav_id: u for u in uavs}
    result = {}

    for u in uavs:
        info = conflict_info[u.uav_id]
        dmin = info["min_dist"]
        neighbors = list(info["conflicts"])

        if dmin >= THR_NEAR_KM or not neighbors:
            result[u.uav_id] = None
            continue

        # متوسط متجه التباعد (Repulsion)
        ax_rep = 0.0
        ay_rep = 0.0

        for nid in neighbors:
            o = id_to_uav[nid]

            dx = u.x - o.x
            dy = u.y - o.y
            dist = math.hypot(dx, dy) + 1e-6

            ax_rep += dx / dist
            ay_rep += dy / dist

        ax_rep /= len(neighbors)
        ay_rep /= len(neighbors)

        # نربط قوة التباعد بالمنطقة (Collision / Inner / Outer)
        if dmin < THR_COLLISION_KM:
            base_scale = 0.015
            note = "collision_zone"
        elif dmin < INNER_NEAR_KM:
            base_scale = 0.008
            note = "inner_near_zone"
        else:
            base_scale = 0.003
            note = "outer_near_zone"

        # تعديل بسيط حسب السرعة الحالية (كلما أسرع → تباعد أكبر)
        speed = u.velocity or 0.0
        speed_factor = 1.0 + min(speed * 50.0, 2.0)  # حد علوي

        scale = base_scale * speed_factor

        suggested_dx = ax_rep * scale
        suggested_dy = ay_rep * scale

        # نقترح تغيير بسيط في السرعة والزاوية (اختياري للمحاكاة)
        new_heading_vec_x = math.cos(u.heading or 0.0) + ax_rep
        new_heading_vec_y = math.sin(u.heading or 0.0) + ay_rep
        new_heading = math.atan2(new_heading_vec_y, new_heading_vec_x)

        dheading = new_heading - (u.heading or 0.0)

        # تقليل السرعة قليلاً في المناطق الحرجة
        if dmin < INNER_NEAR_KM:
            dv = -0.0003
        else:
            dv = -0.0001

        result[u.uav_id] = Avoidance(
            suggested_dx=suggested_dx,
            suggested_dy=suggested_dy,
            suggested_dv=dv,
            suggested_dheading=dheading,
            note=note
        )

    return result


# ======================================
# PUT — UPDATE ONLY
# ======================================
@app.put("/uav")
def put_uav(uav: UAVIn, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc)

    state = db.query(UAVState).filter(UAVState.uav_id == uav.uav_id).first()
    if state is None:
        raise HTTPException(
            status_code=400,
            detail=f"uav_id {uav.uav_id} not initialized. Use /init or /init_random"
        )

    # UPDATE state
    state.x = uav.x
    state.y = uav.y
    state.altitude = uav.altitude
    state.city = uav.city

    state.velocity = float(uav.velocity or 0.0)
    state.heading = float(uav.heading or 0.0)
    state.ax = float(uav.ax or 0.0)
    state.ay = float(uav.ay or 0.0)

    state.timestamp = now

    # HISTORY
    hist = UAVHistory(
        uav_id=uav.uav_id,
        city=uav.city,
        x=uav.x,
        y=uav.y,
        altitude=uav.altitude,
        velocity=float(uav.velocity or 0.0),
        heading=float(uav.heading or 0.0),
        ax=float(uav.ax or 0.0),
        ay=float(uav.ay or 0.0),
        timestamp=now
    )
    db.add(hist)

    db.commit()
    return {"status": "updated", "uav_id": uav.uav_id}


# ======================================
# GET UAV LIST
# ======================================
@app.get("/uavs", response_model=UAVListOut)
def get_uavs(process: bool = False, db: Session = Depends(get_db)):
    uavs = db.query(UAVState).order_by(UAVState.uav_id).all()
    if not uavs:
        return UAVListOut(count=0, uavs=[], collisions=0, near=0, safe=0)

    # حساب الوضع الحالي
    conflict_info = compute_conflicts(uavs)

    # إذا نريد السيرفر يطبّق Avoidance بنفسه
    if process:
        avoidance = compute_server_avoidance(uavs, conflict_info)
        now = datetime.now(timezone.utc)

        for u in uavs:
            a = avoidance.get(u.uav_id)
            if a:
                # نطبّق الإزاحات المقترحة
                u.x += a.suggested_dx
                u.y += a.suggested_dy

                # نحدّث السرعة والزاوية (اختياري – يعتمد على المحاكاة)
                u.velocity = max(0.0, (u.velocity or 0.0) + a.suggested_dv)
                u.heading = (u.heading or 0.0) + a.suggested_dheading

                u.timestamp = now

                db.add(
                    UAVHistory(
                        uav_id=u.uav_id,
                        city=u.city,
                        x=u.x,
                        y=u.y,
                        altitude=u.altitude,
                        velocity=u.velocity,
                        heading=u.heading,
                        ax=u.ax,
                        ay=u.ay,
                        timestamp=now
                    )
                )

        db.commit()
        uavs = db.query(UAVState).order_by(UAVState.uav_id).all()
        conflict_info = compute_conflicts(uavs)
    else:
        avoidance = {u.uav_id: None for u in uavs}

    counts = {"collision": 0, "inner_near": 0, "outer_near": 0, "safe": 0}
    out_list: List[UAVOut] = []

    for u in uavs:
        info = conflict_info[u.uav_id]
        status = info["status"]
        counts[status] += 1

        # تنبؤ قصير المدى + مسار مستقبلي
        pred_short = predict_state(u, 5.0)
        future_path = predict_future_path(u, [5.0, 10.0, 20.0])

        out_list.append(
            UAVOut(
                uav_id=u.uav_id,
                city=u.city,
                x=u.x,
                y=u.y,
                altitude=u.altitude,
                timestamp=u.timestamp,
                velocity=u.velocity or 0.0,
                heading=u.heading or 0.0,
                ax=u.ax or 0.0,
                ay=u.ay or 0.0,
                status=status,
                min_distance_km=round(info["min_dist"], 3),
                predicted=pred_short,
                future_path=future_path,
                avoidance=avoidance.get(u.uav_id),
                conflicts_with=list(info["conflicts"])
            )
        )

    near_total = counts["inner_near"] + counts["outer_near"]

    return UAVListOut(
        count=len(out_list),
        uavs=out_list,
        collisions=counts["collision"],
        near=near_total,
        safe=counts["safe"]
    )


# ======================================
# LOGS
# ======================================
@app.get("/logs")
def get_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = (
        db.query(UAVHistory)
        .order_by(UAVHistory.timestamp.desc())
        .limit(min(limit, 1000))
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
            "velocity": row.velocity,
            "heading": row.heading,
            "ax": row.ax,
            "ay": row.ay,
            "timestamp": row.timestamp,
        }
        for row in logs
    ]


# ======================================
# HEALTH
# ======================================
@app.get("/health")
def health():
    return {"status": "ok"}
