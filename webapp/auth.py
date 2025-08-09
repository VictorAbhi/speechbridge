from database import DatabaseManager

class AuthHandler:
    def __init__(self, db_path="instance/users.db"):
        self.db = DatabaseManager(db_path)

    def register(self, username, password):
        return self.db.add_user(username, password)

    def login(self, username, password):
        return self.db.verify_user(username, password)
