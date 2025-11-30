# ============================================================
# Simple UAV Server – Send & Receive (FastAPI + In-Memory DB)
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------
# Create FastAPI app
# --------------------------------------------
app = FastAPI(title="Simple UAV Server", version="1.0")

# قاعدة بيانات بسيطة داخل الذاكرة (بدون SQLite)
uav_db = {}   # key = uav_id   /   value = dict


# --------------------------------------------
# Pydantic model (شكل بيانات UAV)
# --------------------------------------------
class UAV(BaseModel):
    uav_id: int
    x: float
    y: float
    altitude: float


# --------------------------------------------
# 1) Health Check
# --------------------------------------------
@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Server is running"}


# --------------------------------------------
# 2) PUT /uav  → استلام UAV و تخزينها
# --------------------------------------------
@app.put("/uav")
def put_uav(data: UAV):
    uav_db[data.uav_id] = data.dict()
    return {"status": "stored", "uav_id": data.uav_id}


# --------------------------------------------
# 3) GET /uavs  → إرجاع كل الطائرات
# --------------------------------------------
@app.get("/uavs")
def get_uavs():
    return {
        "count": len(uav_db),
        "uavs": list(uav_db.values())
    }


# --------------------------------------------
# 4) لتشغيل السيرفر محلياً
#    (Render يتجاهل هذا ويستخدم Start Command)
# --------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
