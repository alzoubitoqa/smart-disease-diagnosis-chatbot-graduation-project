from src.core.database import get_connection


def get_profile(user_id: int):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT user_id, age, gender, medical_history FROM profiles WHERE user_id = ?",
        (user_id,)
    )
    profile = cursor.fetchone()
    conn.close()

    if not profile:
        return {
            "user_id": user_id,
            "age": None,
            "gender": None,
            "medical_history": None
        }

    return {
        "user_id": profile["user_id"],
        "age": profile["age"],
        "gender": profile["gender"],
        "medical_history": profile["medical_history"]
    }


def upsert_profile(user_id: int, payload):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM profiles WHERE user_id = ?", (user_id,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("""
            UPDATE profiles
            SET age = ?, gender = ?, medical_history = ?
            WHERE user_id = ?
        """, (payload.age, payload.gender, payload.medical_history, user_id))
    else:
        cursor.execute("""
            INSERT INTO profiles (user_id, age, gender, medical_history)
            VALUES (?, ?, ?, ?)
        """, (user_id, payload.age, payload.gender, payload.medical_history))

    conn.commit()
    conn.close()

    return {
        "user_id": user_id,
        "age": payload.age,
        "gender": payload.gender,
        "medical_history": payload.medical_history
    }