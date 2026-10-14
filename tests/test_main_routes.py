from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_root_redirects_to_the_dashboard():
    # Regression for a real, confirmed-live gap: the dashboard is mounted at
    # /ui, not "/", so a deployed app's bare root URL had no route at all
    # and 404'd — which looked like "the site only shows Swagger," since
    # /docs was the one thing that did resolve.
    # No `with`: intentionally not triggering the app's lifespan, since this
    # route never touches any app.state resource.
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_root_is_excluded_from_the_openapi_schema():
    # A redirect isn't a real API endpoint — it shouldn't clutter /docs.
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    assert "/" not in schema["paths"]
