class UserListFetchError(Exception):
    def __init__(self, message=None, username=None, default_message="", status=None):
        super().__init__(message if message else default_message)
        self.username = username if username else ""
        self.message = self.args[0] if self.args[0] else "List could not be fetched due to server error." \
                                                         " Please try again later."
        self.status = status if status else 500


class UserDoesNotExistError(UserListFetchError):
    def __init__(self, message=None, username=None):
        super().__init__(message, username, "User does not exist.", 404)


class UserListPrivateError(UserListFetchError):
    def __init__(self, message=None, username=None):
        super().__init__(message, username, "This user's list is private.", 403)


class BadListRequest(UserListFetchError):
    def __init__(self, message=None, username=None):
        super().__init__(message, username, "Bad Request", 400)

