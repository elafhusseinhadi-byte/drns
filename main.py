# =====================================================
# 🚀 UAV Single-City Simulation Server – Baghdad Only
#    Prediction-Ready + NFZ + Before/After Prediction
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table
from sqlalchemy.orm import sessionmaker
import random, math, asyncio

# =====================================================
# CITY CONFIG
# =====================================================
CITY_NAME   = "Baghdad"
CITY_CENTER = (33.3, 44.4)

COLLISION_THRESHOLD = 0.05    # نفس الـ Threshold في الواجهة
NEAR_FACTOR         = 2.0     # near = d < TH * NEAR_FACTOR
AVOID_ALT_STEP      = 5       # رفع الارتفاع عند الخطر
AVOID_SPEED_STEP    = -3      # تقليل السرعة عند الاقتراب
MOVE_DT             = 0.002   # خطوة الحركة (Δt)

# مناطق الحظر (NFZ) – تستخدم هنا ومع /uavs
NO_FLY_ZONES = [
    {"cx": 33.3, "cy": 44.4, "r": 0.2},   # Center NFZ
    {"cx": 33.0, "cy": 44.0, "r": 0.15},  # South NFZ
]

# =====================================================
# MODELS
# =====================================================
class UAV(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    speed: float
    system_case: str = "normal"

# =====================================================
# DATABASE SETUP
# =====================================================
engine = create_engine(
    "sqlite:///uav_single_city.sqlite",
    connect_args={"check_same_thread": False}
)

metadata = MetaData()
uav_table = Table(
    "uavs", metadata,
    Column("uav_id", Integer, primary_key=True),
    Column("x", Float),
    Column("y", Float),
    Column("altitude", Float),
    Column("speed", Float),
    Column("direction", Float),
    Column("system_case", String),
)
metadata.create_all(engine)

SessionLocal = sessionmaker(bind=engine)

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="UAV – Single-City Baghdad Simulation (Prediction-Ready)")

@app.get("/")
async def home():
    return {"server": "running", "city": CITY_NAME}

@app.get("/health")
async def health():
    return {"status": "ok"}

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def distance(u1, u2):
    """المسافة الإقليدية بين طائرتين."""
    return math.sqrt((u1.x - u2.x)**2 + (u1.y - u2.y)**2)

def move_uav(u):
    """حركة بسيطة + noise خفيف."""
    x = u.x + u.speed * math.cos(u.direction) * MOVE_DT
    y = u.y + u.speed * math.sin(u.direction) * MOVE_DT

    x += random.uniform(-0.0005, 0.0005)
    y += random.uniform(-0.0005, 0.0005)
    return x, y

def inside_nfz_xy(x, y):
    """هل النقطة (x,y) داخل أي منطقة NFZ؟ ترجع (bool, idx أو None)."""
    for idx, z in enumerate(NO_FLY_ZONES):
        dx = x - z["cx"]
        dy = y - z["cy"]
        if math.sqrt(dx*dx + dy*dy) <= z["r"]:
            return True, idx
    return False, None

def compute_future_prediction(uavs):
    """
    AI-like prediction:
    - نحسب موقع مستقبلي صغير لكل UAV باستخدام direction + speed.
    - إذا المسافة المستقبلية أقل من threshold → نعتبرهم future-risky.
    """
    future_points = []
    future_set = set()

    for u in uavs:
        angle = u.direction
        fx = u.x + u.speed * math.cos(angle) * 0.0001
        fy = u.y + u.speed * math.sin(angle) * 0.0001
        future_points.append((u.uav_id, fx, fy))

    for i in range(len(future_points)):
        for j in range(i+1, len(future_points)):
            uid1, x1, y1 = future_points[i]
            uid2, x2, y2 = future_points[j]
            d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if d < COLLISION_THRESHOLD:
                future_set.add(uid1)
                future_set.add(uid2)

    return future_set

def compute_risk(uavs, future_ids):
    """
    Risk score بسيط لكل UAV:
    - أقل مسافة مع أقرب طائرة
    - حالة near
    - حالة future_risky
    """
    risks = {}
    for u in uavs:
        min_dist = 999.0
        for v in uavs:
            if v.uav_id == u.uav_id:
                continue
            d = distance(u, v)
            if d < min_dist:
                min_dist = d

        is_near   = min_dist < COLLISION_THRESHOLD * NEAR_FACTOR
        is_future = u.uav_id in future_ids

        score = 0.0
        if min_dist < COLLISION_THRESHOLD:
            score += 1.0
        if is_near:
            score += 0.5
        if is_future:
            score += 0.8

        risks[u.uav_id] = round(score, 3)
    return risks

# =====================================================
# PUT /uav – إدخال أو تحديث طائرة
# =====================================================
@app.put("/uav")
async def put_uav(data: UAV):
    session = SessionLocal()
    try:
        existing = session.query(uav_table).filter_by(uav_id=data.uav_id).first()

        # اتجاه عشوائي إذا كانت طائرة جديدة
        direction = random.uniform(0, 2 * math.pi)

        values = {
            "x": data.x,
            "y": data.y,
            "altitude": data.altitude,
            "speed": data.speed,
            "system_case": data.system_case,
            "direction": direction if not existing else existing.direction,
        }

        if existing:
            stmt = uav_table.update().where(
                uav_table.c.uav_id == data.uav_id
            ).values(**values)
        else:
            stmt = uav_table.insert().values(uav_id=data.uav_id, **values)

        session.execute(stmt)
        session.commit()
        return {"status": "ok"}

    finally:
        session.close()

# =====================================================
# GET /uavs – إرجاع حالة الطائرات مع risk + future_risk + NFZ flag
# =====================================================
@app.get("/uavs")
async def get_uavs():
    session = SessionLocal()
    try:
        rows = session.query(uav_table).all()
        if not rows:
            return {"uavs": []}

        uavs = rows
        future_ids = compute_future_prediction(uavs)
        risks = compute_risk(uavs, future_ids)

        result = []
        for u in uavs:
            in_nfz, _ = inside_nfz_xy(u.x, u.y)
            result.append({
                "uav_id": u.uav_id,
                "x": u.x,
                "y": u.y,
                "altitude": u.altitude,
                "speed": u.speed,
                "direction": u.direction,
                "system_case": u.system_case,
                "future_risk": (u.uav_id in future_ids),
                "risk_score": risks[u.uav_id],
                "in_nfz": in_nfz,
            })

        return {"uavs": result}

    finally:
        session.close()

# =====================================================
# DELETE /reset – مسح كل الطائرات
# =====================================================
@app.delete("/reset")
async def reset():
    session = SessionLocal()
    try:
        session.query(uav_table).delete()
        session.commit()
        return {"status": "cleared"}
    finally:
        session.close()

# =====================================================
# POST /process – الحركة + NFZ + Prediction Before/After
# =====================================================
@app.post("/process")
async def process_step():
    session = SessionLocal()
    try:
        uavs = session.query(uav_table).all()
        n = len(uavs)

        if n == 0:
            return {
                "processed": 0,
                "collisions_before_prediction": 0,
                "near_before_prediction": 0,
                "collisions_after_prediction": 0,
                "near_after_prediction": 0,
                "nfz_hits": 0,
                "future_risky": [],
                "risk_scores": {},
                "status": "ok"
            }

        # --------------------------------------------
        # (0) Enforcement NFZ أولاً (بيئة ثابتة)
        # --------------------------------------------
        nfz_hits = 0
        for u in uavs:
            in_nfz, idx = inside_nfz_xy(u.x, u.y)
            if in_nfz:
                nfz_hits += 1
                z = NO_FLY_ZONES[idx]
                dx = u.x - z["cx"]
                dy = u.y - z["cy"]
                d  = math.sqrt(dx*dx + dy*dy) or 0.001
                # نخرجه إلى خارج الدائرة بشعاع بسيط
                u.x = z["cx"] + dx/d * (z["r"] + 0.01)
                u.y = z["cy"] + dy/d * (z["r"] + 0.01)
                # نرفع الارتفاع ونقلل السرعة (سلوك أمان)
                u.altitude += AVOID_ALT_STEP
                u.speed = max(2, u.speed + AVOID_SPEED_STEP)

        # --------------------------------------------
        # (1) Collisions & Near "قبل التنبؤ"
        # --------------------------------------------
        coll_before_pred = 0
        near_before_pred = 0
        for i in range(n):
            for j in range(i+1, n):
                d = distance(uavs[i], uavs[j])
                if d < COLLISION_THRESHOLD:
                    coll_before_pred += 1
                elif d < COLLISION_THRESHOLD * NEAR_FACTOR:
                    near_before_pred += 1

        # --------------------------------------------
        # (2) حساب التنبؤ المستقبلي (AI-like)
        # --------------------------------------------
        future_ids = compute_future_prediction(uavs)

        # --------------------------------------------
        # (3) تطبيق خوارزمية التنبؤ (Prediction Avoidance)
        #     فقط على الطائرات الـ future_risky
        # --------------------------------------------
        for u in uavs:
            if u.uav_id in future_ids:
                # نرفع الارتفاع
                u.altitude += AVOID_ALT_STEP
                # نبطئ السرعة
                u.speed = max(2, u.speed + AVOID_SPEED_STEP)
                # نغير الاتجاه قليلاً
                u.direction += random.uniform(-0.3, 0.3)

        # --------------------------------------------
        # (4) Collisions & Near "بعد التنبؤ"
        # --------------------------------------------
        coll_after_pred = 0
        near_after_pred = 0
        for i in range(n):
            for j in range(i+1, n):
                d = distance(uavs[i], uavs[j])
                if d < COLLISION_THRESHOLD:
                    coll_after_pred += 1
                elif d < COLLISION_THRESHOLD * NEAR_FACTOR:
                    near_after_pred += 1
                    # ممكن أيضاً نطبق avoidance عادي هنا لو حابة
                    # لكن حالياً نكتفي بحساب الإحصائيات

        # --------------------------------------------
        # (5) حركة الطائرات بعد كل التعديلات
        # --------------------------------------------
        for u in uavs:
            new_x, new_y = move_uav(u)
            stmt = uav_table.update().where(
                uav_table.c.uav_id == u.uav_id
            ).values(
                x=new_x,
                y=new_y,
                altitude=u.altitude,
                speed=u.speed,
                direction=u.direction,
            )
            session.execute(stmt)

        session.commit()

        # --------------------------------------------
        # (6) Risk + Future بعد الحركة
        # --------------------------------------------
        # نعيد قراءة الحالات حتى نحسب risk على مواقع شبه محدثة
        uavs_after = session.query(uav_table).all()
        future_ids_after = compute_future_prediction(uavs_after)
        risks = compute_risk(uavs_after, future_ids_after)

        await asyncio.sleep(0.001 * n)

        return {
            "processed": n,
            # قيم خاصة بالداشبورد (قبل/بعد التنبؤ)
            "collisions_before_prediction": coll_before_pred,
            "near_before_prediction": near_before_pred,
            "collisions_after_prediction": coll_after_pred,
            "near_after_prediction": near_after_pred,
            # معلومات إضافية (تقدرين تستخدمينها إذا حبيتي)
            "nfz_hits": nfz_hits,
            "future_risky": list(future_ids_after),
            "risk_scores": risks,
            "status": "ok"
        }

    finally:
        session.close()

# =====================================================
# LOCAL RUN (for testing)
# =====================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
