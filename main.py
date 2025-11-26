# =====================================================
# 🛰 UAV Baghdad Server – AI Path + Collision Avoidance
#   - Single City: Baghdad Only
#   - FastAPI + SQLite + SQLAlchemy (ORM)
#   - Server computes direction + goal (AI path)
#   - Server performs collision avoidance
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker, declarative_base
from math import sqrt, atan2, cos, sin, pi
import random

# -----------------------------------------------------
# ⚙️ Simulation constants
# -----------------------------------------------------
BAGHDAD_CENTER_X = 33.3
BAGHDAD_CENTER_Y = 44.4

# حدود تقريبية للحركة داخل بغداد
BAGHDAD_X_MIN = 33.0
BAGHDAD_X_MAX = 33.6
BAGHDAD_Y_MIN = 44.1
BAGHDAD_Y_MAX = 44.7

COLLISION_THRESHOLD = 0.05   # threshold (approx degrees) for collision
NEAR_FACTOR         = 2.0    # near = COLLISION_THRESHOLD * NEAR_FACTOR
DT                   = 1.0   # simulation time step (sec)
SCALE                = 0.0001  # تحويل m/s إلى تحرك بالإحداثيات (تقريبياً)

# -----------------------------------------------------
# 🛢️ Database setup (SQLite)
# -----------------------------------------------------
DB_URL = "sqlite:///./uav_baghdad.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class UAVModel(Base):
    __tablename__ = "uavs"

    uav_id     = Column(Integer, primary_key=True, index=True)
    x          = Column(Float, nullable=False)
    y          = Column(Float, nullable=False)
    altitude   = Column(Float, nullable=False)
    speed      = Column(Float, nullable=False)
    direction  = Column(Float, default=0.0)      # راديان
    system_case = Column(String, default="normal")

    # AI path: هدف داخلي لكل UAV
    goal_x     = Column(Float, nullable=True)
    goal_y     = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

# -----------------------------------------------------
# 📦 FastAPI app
# -----------------------------------------------------
app = FastAPI(
    title="Baghdad UAV Server – AI Path + Collision Avoidance",
    version="1.0"
)

# -----------------------------------------------------
# 📨 Pydantic models
# -----------------------------------------------------
class UAVIn(BaseModel):
    """شكل البيانات اللي يرسلها الـ Client بالسيرفر."""
    uav_id: int
    x: float
    y: float
    altitude: float
    speed: float
    system_case: str = "normal"
    # اختياري: إذا بديتي بديريكشن من الـ client
    direction: Optional[float] = None

class UAVOut(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    speed: float
    direction: float
    system_case: str
    goal_x: Optional[float]
    goal_y: Optional[float]

# -----------------------------------------------------
# 🧩 Helper: random goal inside Baghdad
# -----------------------------------------------------
def random_goal_inside_baghdad():
    gx = random.uniform(BAGHDAD_X_MIN, BAGHDAD_X_MAX)
    gy = random.uniform(BAGHDAD_Y_MIN, BAGHDAD_Y_MAX)
    return gx, gy

# -----------------------------------------------------
# 🧩 Helper: distance
# -----------------------------------------------------
def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# -----------------------------------------------------
# 🔁 /reset – حذف كل الـ UAVs
# -----------------------------------------------------
@app.delete("/reset")
def reset():
    db = SessionLocal()
    try:
        db.query(UAVModel).delete()
        db.commit()
        return {"status": "reset_done", "city": "Baghdad"}
    finally:
        db.close()

# -----------------------------------------------------
# 📥 PUT /uav – إدخال أو تحديث UAV
#   Client يحط بس Baghdad, السيرفر يتكفل بالـ direction + goal
# -----------------------------------------------------
@app.put("/uav")
def upsert_uav(uav: UAVIn):
    db = SessionLocal()
    try:
        row = db.query(UAVModel).filter(UAVModel.uav_id == uav.uav_id).first()

        if row is None:
            # أول مرة – نعين هدف عشوائي داخل بغداد
            gx, gy = random_goal_inside_baghdad()
            direction = uav.direction if uav.direction is not None else random.uniform(-pi, pi)

            row = UAVModel(
                uav_id=uav.uav_id,
                x=uav.x,
                y=uav.y,
                altitude=uav.altitude,
                speed=uav.speed,
                direction=direction,
                system_case=uav.system_case,
                goal_x=gx,
                goal_y=gy,
            )
            db.add(row)
        else:
            # تحديث – نبقي الهدف إذا موجود
            row.x = uav.x
            row.y = uav.y
            row.altitude = uav.altitude
            row.speed = uav.speed
            row.system_case = uav.system_case

            if uav.direction is not None:
                row.direction = uav.direction

            # إذا ما عنده هدف – نعين هدف جديد
            if row.goal_x is None or row.goal_y is None:
                gx, gy = random_goal_inside_baghdad()
                row.goal_x, row.goal_y = gx, gy

        db.commit()
        return {"status": "ok", "uav_id": uav.uav_id}
    finally:
        db.close()

# -----------------------------------------------------
# 📤 GET /uavs – إرجاع كل الطائرات
# -----------------------------------------------------
@app.get("/uavs", response_model=dict)
def get_uavs():
    db = SessionLocal()
    try:
        rows = db.query(UAVModel).all()
        out: List[UAVOut] = []
        for r in rows:
            out.append(UAVOut(
                uav_id=r.uav_id,
                x=r.x,
                y=r.y,
                altitude=r.altitude,
                speed=r.speed,
                direction=r.direction,
                system_case=r.system_case,
                goal_x=r.goal_x,
                goal_y=r.goal_y,
            ))
        return {"count": len(out), "uavs": [o.dict() for o in out]}
    finally:
        db.close()

# -----------------------------------------------------
# 🤖 AI Path + Collision Avoidance – /process
#   - يحسب اتجاه لكل UAV نحو goal
#   - يحرك الطائرات خطوة واحدة
#   - يكشف الأزواج القريبة / المتصادمة
#   - ينفذ Collision Avoidance (تغيير اتجاه + تقليل سرعة)
# -----------------------------------------------------
@app.post("/process")
def process_step():
    db = SessionLocal()
    try:
        uavs: List[UAVModel] = db.query(UAVModel).all()
        if not uavs:
            return {"status": "no_uavs"}

        # 1) تأكد لكل UAV هدف (goal_x, goal_y)
        for u in uavs:
            if u.goal_x is None or u.goal_y is None:
                u.goal_x, u.goal_y = random_goal_inside_baghdad()

        # 2) احسب الـ base direction ناحية الهدف
        for u in uavs:
            dxg = u.goal_x - u.x
            dyg = u.goal_y - u.y
            # إذا الهدف قريب جداً – عيّن هدف جديد حتى تستمر الحركة
            if dist(u.x, u.y, u.goal_x, u.goal_y) < 0.02:
                u.goal_x, u.goal_y = random_goal_inside_baghdad()
                dxg = u.goal_x - u.x
                dyg = u.goal_y - u.y
            u.direction = atan2(dyg, dxg)
            # نرجع system_case طبيعي قبل الحساب
            u.system_case = "normal"

        # 3) احسب موضع مقترح (proposed positions) بدون Avoidance
        proposed = {}
        for u in uavs:
            nx = u.x + u.speed * DT * SCALE * cos(u.direction)
            ny = u.y + u.speed * DT * SCALE * sin(u.direction)
            proposed[u.uav_id] = (nx, ny)

        # 4) كشف الاصطدام / القرب
        collision_pairs = set()    # أزواج مسافة < COLLISION_THRESHOLD
        near_pairs = set()         # أزواج مسافة < NEAR_FACTOR * COLLISION_THRESHOLD

        for i in range(len(uavs)):
            ui = uavs[i]
            xi, yi = proposed[ui.uav_id]
            for j in range(i + 1, len(uavs)):
                uj = uavs[j]
                xj, yj = proposed[uj.uav_id]
                d = dist(xi, yi, xj, yj)
                if d < COLLISION_THRESHOLD:
                    collision_pairs.add(frozenset({ui.uav_id, uj.uav_id}))
                elif d < COLLISION_THRESHOLD * NEAR_FACTOR:
                    near_pairs.add(frozenset({ui.uav_id, uj.uav_id}))

        # 5) Collision Avoidance: عدّل الاتجاه + السرعة للأزواج الخطرة
        # نستخدم قاموس لسهولة الوصول
        uav_by_id = {u.uav_id: u for u in uavs}

        # أزواج الاصطدام – مناورة قوية
        for pair in collision_pairs:
            id1, id2 = tuple(pair)
            u1 = uav_by_id[id1]
            u2 = uav_by_id[id2]

            # اتجاه الخط بينهما
            angle_12 = atan2(u2.y - u1.y, u2.x - u1.x)

            # نلفهم ± 90 درجة حتى يبتعدون
            turn_angle = pi / 2.0

            u1.direction = angle_12 - turn_angle
            u2.direction = angle_12 + turn_angle

            # نقلل السرعة شوية لزيادة الأمان
            u1.speed *= 0.7
            u2.speed *= 0.7

            u1.system_case = "avoidance"
            u2.system_case = "avoidance"

        # أزواج القريبة فقط – مناورة خفيفة
        for pair in near_pairs:
            if pair in collision_pairs:
                continue
            id1, id2 = tuple(pair)
            u1 = uav_by_id[id1]
            u2 = uav_by_id[id2]

            angle_12 = atan2(u2.y - u1.y, u2.x - u1.x)
            turn_angle = pi / 4.0  # 45 درجة

            # نغيّر الاتجاه شويّة بعيداً عن بعض
            u1.direction = angle_12 - turn_angle
            u2.direction = angle_12 + turn_angle

            # تقليل بسيط بالسرعة
            u1.speed *= 0.9
            u2.speed *= 0.9

            # نخليها "avoidance" حتى يميّزها الـ Dashboard
            if u1.system_case != "avoidance":
                u1.system_case = "avoidance"
            if u2.system_case != "avoidance":
                u2.system_case = "avoidance"

        # 6) احسب الموضع النهائي بعد Collision Avoidance
        moved = 0
        for u in uavs:
            nx = u.x + u.speed * DT * SCALE * cos(u.direction)
            ny = u.y + u.speed * DT * SCALE * sin(u.direction)

            # نضمن تبقى داخل حدود بغداد التقريبية
            nx = min(max(nx, BAGHDAD_X_MIN), BAGHDAD_X_MAX)
            ny = min(max(ny, BAGHDAD_Y_MIN), BAGHDAD_Y_MAX)

            u.x = nx
            u.y = ny
            moved += 1

        db.commit()

        return {
            "status": "ok",
            "processed": moved,
            "collisions_detected": len(collision_pairs),
            "near_pairs": len(near_pairs),
            "collision_pairs": [list(p) for p in collision_pairs],
            "near_pairs_list": [list(p) for p in near_pairs],
        }
    finally:
        db.close()

# -----------------------------------------------------
# 🌐 Root
# -----------------------------------------------------
@app.get("/")
def root():
    return {
        "server": "Baghdad UAV Server – AI Path + Collision Avoidance",
        "city": "Baghdad",
        "note": "Use /uav (PUT), /uavs (GET), /process (POST), /reset (DELETE)."
    }
