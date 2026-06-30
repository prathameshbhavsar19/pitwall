"""
Locust load test for PitWall /query endpoint.

Run with:
    locust -f locustfile.py --host http://127.0.0.1:8000

Then open http://localhost:8089 in your browser to configure
number of users, spawn rate, and start the test.
"""

from locust import HttpUser, task, between
import random

# A pool of realistic queries — locust picks one at random per request
# so we're not just hammering the cache with the identical query every time
QUERIES = [
    "What is the penalty for unsafe release during a race?",
    "What is the maximum number of power units a manufacturer can use?",
    "What are the DRS activation zone rules?",
    "Under what circumstances can the Race Director close the pit lane?",
    "What is the software debounce limit for electronic systems?",
    "What fine may be imposed if a driver retires due to unsafe release?",
    "What restrictions apply to public statements between teams?",
    "What is required of the isolation monitoring system?",
    "What penalty applies if a driver enters a closed pit lane?",
    "What are the conditions for a component to be considered available?",
]


class PitWallUser(HttpUser):
    # Each simulated user waits 1-3 seconds between requests,
    # mimicking a real person reading the answer before asking again
    wait_time = between(1, 3)

    @task
    def query_endpoint(self):
        query_text = random.choice(QUERIES)
        self.client.post(
            "/query",
            json={"query": query_text, "top_k": 3},
            name="/query",  # groups all variations under one stat row in the UI
        )