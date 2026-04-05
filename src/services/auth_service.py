from src.core.database import get_connection
from src.core.security import hash_password, verify_password, create_access_token


def register_user(payload):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE email = ?", (payload.email,))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        raise ValueError("Email already registered.")

    password_hash = hash_password(payload.password)

    cursor.execute(
        "INSERT INTO users (email, password_hash) VALUES (?, ?)",
        (payload.email, password_hash)
    )
    conn.commit()

    user_id = cursor.lastrowid

    cursor.execute(
        "INSERT OR IGNORE INTO profiles (user_id, age, gender, medical_history) VALUES (?, ?, ?, ?)",
        (user_id, None, None, None)
    )
    conn.commit()
    conn.close()

    token = create_access_token({"sub": str(user_id), "email": payload.email})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_id,
        "email": payload.email
    }


def login_user(payload):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id, email, password_hash FROM users WHERE email = ?",
        (payload.email,)
    )
    user = cursor.fetchone()
    conn.close()

    if not user:
        raise ValueError("Invalid email or password.")

    if not verify_password(payload.password, user["password_hash"]):
        raise ValueError("Invalid email or password.")

    token = create_access_token({"sub": str(user["id"]), "email": user["email"]})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user["id"],
        "email": user["email"]
    }