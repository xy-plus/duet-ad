import hmac

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


async def require_auth(
    request: Request,
    cred: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    expected = request.app.state.settings.access_token
    if cred is None or not hmac.compare_digest(cred.credentials, expected):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "unauthorized",
                "message": "登录凭据无效或已过期",
            },
        )
