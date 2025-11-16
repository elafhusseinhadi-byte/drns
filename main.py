# =====================================================
# 🚀 UAV Simulation Server (Online Ready) - Multi-City + Transfer
# =====================================================
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, Float, String, MetaData, Table, and_
from sqlalchemy.orm import sessionmaker
import time, asyncio

# -------------------------------
# 🌍 تعريف إحداثيات المدن (تقديرية)
# تگدرين تزيدين مدن أكثر بكل بساطة
# -------------------------------
CITY_COORDS = {
    "Baghdad": (33.3, 44.4),
    "Basra":   (30.5, 47.8),
    "Najaf":   (31.99, 44.31),
}

# -------------------------------
# 🛰️ نموذج بيانات UAV من الـ Client
# -------------------------------
class UAV(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float
    speed: float
    system_case: str  # normal, avoidance
    # حقول اختيارية لدعم النقل بين المدن
    target_city: str | None = None
    progress: int = 0  # 0..100

# طلب نقل طائرة بين مدينتين
class TransferRequest(BaseModel):
    uav_id: int
    from_city: str
    to_city: str

# -------------------------------
# ⚙️ إعداد قاعدة بيانات SQLite
# -------------------------------
engine = create_engine("sqlite:///uav_db_full.sqlite",
                       connect_args={"check_same_thread": False})
metadata = MetaData()

uav_table = Table(
    "uavs", metadata,
    Column("uav_id", Integer, primary_key=True),
    Column("city_name", String, index=True),  # المدينة الحالية للطائرة
    Column("x", Float),
    Column("y", Float),
    Column("altitude", Float),
    Column("speed", Float),
    Column("system_case", String),
    # 🔴 جديد: المدينة الهدف ونسبة التقدم في الرحلة
    Column("target_city", String, nullable=True),
    Column("progress", Integer, default=0),
)

# ملاحظة: create_all لن يحذف الجدول القديم، فقط يضيف الأعمدة الجديدة إذا الجدول جديد
metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

# -------------------------------
# 🖥️ إعداد FastAPI server
# -------------------------------
app = FastAPI(title="UAV Simulation Server (Online + Multi-City)")

# -------------------------------
# 🛰️ PUT /city/{city}/uav
# تخزين/تحديث بيانات طائرة في مدينة معيّنة
# يدعم أيضًا target_city و progress
# -------------------------------
@app.put("/city/{city}/uav")
async def put_uav(city: str, data: UAV):
    session = SessionLocal()
    start = time.time()
    try:
        existing = session.query(uav_table).filter_by(
            city_name=city,
            uav_id=data.uav_id
        ).first()

        values = {
            "x": data.x,
            "y": data.y,
            "altitude": data.altitude,
            "speed": data.speed,
            "system_case": data.system_case,
            "city_name": city,
            "target_city": data.target_city,
            "progress": data.progress,
        }

        if existing:
            stmt = (
                uav_table.update()
                .where(and_(uav_table.c.city_name == city,
                            uav_table.c.uav_id == data.uav_id))
                .values(**values)
            )
            session.execute(stmt)
        else:
            values["uav_id"] = data.uav_id
            stmt = uav_table.insert().values(**values)
            session.execute(stmt)

        session.commit()
        elapsed_ms = (time.time() - start) * 1000
        return {"status": "ok", "put_time_ms": round(elapsed_ms, 3)}
    finally:
        session.close()

# -------------------------------
# 📦 GET /city/{city}/uavs
# استرجاع كل الطائرات في المدينة (مع حالة اختيارية)
# -------------------------------
@app.get("/city/{city}/uavs")
async def get_uavs(city: str, system_case: str = None):
    session = SessionLocal()
    start = time.time()
    try:
        query = session.query(uav_table).filter_by(city_name=city)
        if system_case:
            query = query.filter_by(system_case=system_case)
        uavs = query.all()

        elapsed_ms = (time.time() - start) * 1000
        approx_db_kb = round(len(uavs) * 0.5, 2)

        return {
            "uavs": [
                {
                    "uav_id": u.uav_id,
                    "x": u.x,
                    "y": u.y,
                    "altitude": u.altitude,
                    "speed": u.speed,
                    "system_case": u.system_case,
                    "city_name": u.city_name,
                    "target_city": u.target_city,
                    "progress": u.progress,
                }
                for u in uavs
            ],
            "get_time_ms": round(elapsed_ms, 3),
            "db_size_kb": approx_db_kb,
        }
    finally:
        session.close()

# -------------------------------
# 🔁 POST /transfer
# بدء عملية نقل طائرة من مدينة إلى أخرى
# -------------------------------
@app.post("/transfer")
async def transfer_uav(req: TransferRequest):
    session = SessionLocal()
    try:
        # نبحث عن الطائرة في المدينة المصدر
        uav = (
            session.query(uav_table)
            .filter_by(city_name=req.from_city, uav_id=req.uav_id)
            .first()
        )
        if not uav:
            return {"status": "error", "message": "UAV not found in source city"}

        # نضبط الهدف ونخلي progress = 0
        stmt = (
            uav_table.update()
            .where(
                and_(
                    uav_table.c.city_name == req.from_city,
                    uav_table.c.uav_id == req.uav_id,
                )
            )
            .values(target_city=req.to_city, progress=0)
        )
        session.execute(stmt)
        session.commit()
        return {
            "status": "ok",
            "message": f"Transfer started from {req.from_city} to {req.to_city}",
        }
    finally:
        session.close()

# -------------------------------
# 🧠 دالة داخلية لتحريك الطائرات بين المدن
# (تُستدعى من /process)
# -------------------------------
def update_transfers(session, city: str):
    """تحديث مواقع الطائرات التي في حالة انتقال بين المدن."""
    # نجيب كل الطائرات في هذه المدينة اللي عندها target_city
    uavs = (
        session.query(uav_table)
        .filter_by(city_name=city)
        .filter(uav_table.c.target_city.isnot(None))
        .all()
    )

    moved = 0

    for u in uavs:
        if u.city_name not in CITY_COORDS or u.target_city not in CITY_COORDS:
            continue

        # نقطة بداية ونقطة نهاية
        Ax, Ay = CITY_COORDS[u.city_name]
        Bx, By = CITY_COORDS[u.target_city]

        # نزيد progress (مثلاً 10% كل مرة معالجة)
        new_progress = min((u.progress or 0) + 10, 100)
        t = new_progress / 100.0

        new_x = Ax + t * (Bx - Ax)
        new_y = Ay + t * (By - Ay)

        # نحدّث الإحداثيات والتقدّم
        stmt = (
            uav_table.update()
            .where(
                and_(
                    uav_table.c.city_name == u.city_name,
                    uav_table.c.uav_id == u.uav_id,
                )
            )
            .values(x=new_x, y=new_y, progress=new_progress)
        )

        # إذا وصلت 100% ننقلها فعلياً للمدينة الهدف
        if new_progress >= 100:
            stmt = stmt.values(
                city_name=u.target_city,
                target_city=None,  # وقفت عملية النقل
            )

        session.execute(stmt)
        moved += 1

    return moved

# -------------------------------
# ⚙️ POST /city/{city}/process
# يحسب التصادمات + يحدّث مسارات النقل بين المدن
# -------------------------------
@app.post("/city/{city}/process")
async def process_uavs(city: str, system_case: str = None):
    session = SessionLocal()
    start = time.time()
    try:
        # أولاً: نحدّث الطائرات اللي في حالة نقل بين المدن
        moved_count = update_transfers(session, city)
        session.commit()

        # ثانياً: نقرأ البيانات بعد التحديث
        query = session.query(uav_table).filter_by(city_name=city)
        if system_case:
            query = query.filter_by(system_case=system_case)
        uavs = query.all()
        n = len(uavs)

        collision_pairs = []

        # 🔍 كشف التصادم (distance < 5 كما في ملفك الأصلي)
        for i in range(n):
            for j in range(i + 1, n):
                dx = uavs[i].x - uavs[j].x
                dy = uavs[i].y - uavs[j].y
                if (dx ** 2 + dy ** 2) ** 0.5 < 5:
                    collision_pairs.append([uavs[i].uav_id, uavs[j].uav_id])

        # محاكاة زمن المعالجة
        await asyncio.sleep(0.001 * n)
        elapsed_ms = (time.time() - start) * 1000
        avg_per_uav = round(elapsed_ms / n, 3) if n > 0 else 0

        return {
            "processed_uavs": n,
            "moved_uavs": moved_count,  # كم طائرة تحركت في هذه الدورة
            "post_time_ms": round(elapsed_ms, 3),
            "avg_post_per_uav_ms": avg_per_uav,
            "collisions_detected": len(collision_pairs),
            "collision_pairs": collision_pairs,
        }
    finally:
        session.close()

# -------------------------------
# ✅ Health Check بسيط (اختياري)
# -------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------------
# 🌍 تشغيل السيرفر محلياً
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
