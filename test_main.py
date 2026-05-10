from fastapi.testclient import TestClient
import pytest
from main import app

# Initialize the test client
client = TestClient(app)

def test_health_endpoint():
    """Test that the health check endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_intent_clarifying():
    """Test that a vague request triggers the CLARIFYING intent."""
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "I need an assessment for a candidate."}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    # Vague requests should not return recommendations yet
    assert len(data["recommendations"]) == 0

def test_intent_ready_to_recommend():
    """Test that a specific request triggers recommendations."""
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "I need an assessment for a software engineer focused on AWS and Cloud."}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    # Specific requests should return recommendations (we know AWS is in the catalog)
    assert len(data["recommendations"]) > 0
    # Check structure of recommendations
    for rec in data["recommendations"]:
        assert "name" in rec
        assert "url" in rec
        assert "test_type" in rec

def test_intent_off_topic():
    """Test that off-topic requests are rejected."""
    response = client.post(
        "/chat",
        json={
            "messages": [
                {"role": "user", "content": "Can you recommend a good Italian restaurant nearby?"}
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    # Should reply with a guardrail message refusing to answer
    assert "only assist" in data["reply"] or "assessments" in data["reply"]
    assert len(data["recommendations"]) == 0

def test_conversation_state_accumulation():
    """
    Test that the model accumulates constraints across multiple turns.
    This simulates a full conversation.
    """
    # Turn 1: User gives role
    # Turn 2: User adds specific skill
    conversation = [
        {"role": "user", "content": "I am hiring a marketing manager."},
        {"role": "assistant", "content": "Great. What specific skills or constraints do they need?"},
        {"role": "user", "content": "They need to know about Agile methods."}
    ]
    
    response = client.post(
        "/chat",
        json={"messages": conversation}
    )
    assert response.status_code == 200
    data = response.json()
    
    # The response should likely contain recommendations related to Agile
    # because the state should have accumulated both "marketing manager" and "Agile"
    assert len(data["recommendations"]) > 0
    
    # At least one recommendation should be related to Agile (which is in the catalog)
    agile_found = False
    for rec in data["recommendations"]:
        if "Agile" in rec["name"] or "Agile" in rec["test_type"]:
            agile_found = True
            break
    assert agile_found, "Agile test was not recommended despite being requested in the conversation history."
