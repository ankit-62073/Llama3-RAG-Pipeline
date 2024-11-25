import sqlite3
import hashlib
from pathlib import Path

class AuthManager:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      password TEXT NOT NULL)''')
        conn.commit()
        conn.close()

    def _hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def register_user(self, username, password):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            hashed_pwd = self._hash_password(password)
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                     (username, hashed_pwd))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def authenticate_user(self, username, password):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        hashed_pwd = self._hash_password(password)
        c.execute("SELECT id FROM users WHERE username=? AND password=?",
                 (username, hashed_pwd))
        user = c.fetchone()
        conn.close()
        return user[0] if user else None