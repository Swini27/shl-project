import json
import os
from typing import List, Dict, Any

CATALOG_FILE = os.path.join(os.path.dirname(__file__), "catalog.json")

class CatalogHelper:
    def __init__(self, filepath: str = CATALOG_FILE):
        self.filepath = filepath
        self.catalog: List[Dict[str, Any]] = self._load_catalog()

    def _load_catalog(self) -> List[Dict[str, Any]]:
        """Loads the JSON catalog from disk."""
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading catalog from {self.filepath}: {e}")
            return []

    def get_all_tests(self) -> List[Dict[str, Any]]:
        """Returns all loaded tests."""
        return self.catalog

    def search_tests(self, query: str) -> List[Dict[str, Any]]:
        """
        A simple keyword-based search for mockup purposes.
        Checks if the query string is in the test's name, job_levels, keys, or description.
        """
        if not query:
            return []
            
        query_lower = query.lower()
        results = []
        for test in self.catalog:
            # Check name
            if query_lower in test.get("name", "").lower():
                results.append(test)
                continue
            
            # Check job levels (seniority/role)
            job_levels = test.get("job_levels", [])
            if any(query_lower in level.lower() for level in job_levels):
                results.append(test)
                continue
                
            # Check keys (categories/skills)
            keys = test.get("keys", [])
            if any(query_lower in key.lower() for key in keys):
                results.append(test)
                continue
                
            # Check description
            if query_lower in test.get("description", "").lower():
                results.append(test)
                continue
                
        return results

    def get_test_by_name(self, name: str) -> Dict[str, Any]:
        """Fetch a specific test by its exact or partial name."""
        name_lower = name.lower()
        for test in self.catalog:
            if name_lower in test.get("name", "").lower():
                return test
        return {}
