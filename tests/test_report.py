import unittest
from lib.report import ReportGenerator
from lib.generate import setupDjango
from bs4 import BeautifulSoup


class TestHTMLReportGeneration(unittest.TestCase):
    """Test HTML report generation for various metric types."""

    def setUp(self):
        """Set up test data for HTML report generation."""
        # Setup Django for templates
        setupDjango()

        # Create test data that includes scalar metrics (crash events)
        self.test_data = {
            "branches": ["control", "treatment"],
            "segments": ["Windows", "Mac"],
            "slug": "test-experiment",
            "startDate": "2024-01-01",
            "endDate": "2024-01-07",
            "channel": "release",
            "isRollout": False,
            "is_experiment": True,
            "control": {
                "Windows": {
                    "numerical": {
                        "fcp_time": {
                            "desc": "Time to first contentful paint",
                            "mean": 1500.5,
                            "median": 1450.0,
                            "n": 1000,
                            "confidence": {"min": 0, "max": 0},
                            "se": 0,
                            "var": 50000,
                            "std": 223.6,
                        }
                    },
                    "categorical": {},
                    "scalar": {
                        "total_crashes": {
                            "desc": "Total count of crashes for this experiment branch",
                            "count": 342,
                            "n": 1,
                        }
                    },
                },
                "Mac": {
                    "numerical": {
                        "fcp_time": {
                            "desc": "Time to first contentful paint",
                            "mean": 1600.2,
                            "median": 1550.0,
                            "n": 800,
                            "confidence": {"min": 0, "max": 0},
                            "se": 0,
                            "var": 60000,
                            "std": 244.9,
                        }
                    },
                    "categorical": {},
                    "scalar": {
                        "total_crashes": {
                            "desc": "Total count of crashes for this experiment branch",
                            "count": 19,
                            "n": 1,
                        }
                    },
                },
            },
            "treatment": {
                "Windows": {
                    "numerical": {
                        "fcp_time": {
                            "desc": "Time to first contentful paint",
                            "mean": 1450.8,
                            "median": 1400.0,
                            "n": 950,
                            "confidence": {"min": 0, "max": 0},
                            "se": 0,
                            "var": 48000,
                            "std": 219.1,
                        }
                    },
                    "categorical": {},
                    "scalar": {
                        "total_crashes": {
                            "desc": "Total count of crashes for this experiment branch",
                            "count": 292,
                            "n": 1,
                            "control_count": 342,
                            "treatment_count": 292,
                            "relative_change": -14.6,
                            "absolute_difference": -50,
                        }
                    },
                },
                "Mac": {
                    "numerical": {
                        "fcp_time": {
                            "desc": "Time to first contentful paint",
                            "mean": 1580.1,
                            "median": 1530.0,
                            "n": 750,
                            "confidence": {"min": 0, "max": 0},
                            "se": 0,
                            "var": 55000,
                            "std": 234.5,
                        }
                    },
                    "categorical": {},
                    "scalar": {
                        "total_crashes": {
                            "desc": "Total count of crashes for this experiment branch",
                            "count": 38,
                            "n": 1,
                            "control_count": 19,
                            "treatment_count": 38,
                            "relative_change": 100.0,
                            "absolute_difference": 19,
                        }
                    },
                },
            },
            "histograms": {},
            "pageload_event_metrics": {
                "fcp_time": {
                    "desc": "Time to first contentful paint",
                    "min": 0,
                    "max": 30000,
                    "kind": "numerical",
                }
            },
            "crash_event_metrics": {
                "total_crashes": {
                    "desc": "Total count of crashes for this experiment branch",
                    "kind": "scalar",
                }
            },
            "input": {
                "branches": [
                    {"name": "control"},
                    {"name": "treatment"},
                ],
                "segments": ["Windows", "Mac"],
            },
            "queries": [
                {
                    "name": "Test Query - Total Crashes",
                    "query": "SELECT branch, count(*) as crash_count FROM test_table GROUP BY branch",
                }
            ],
        }

    def test_scalar_metrics_generate_canvas_and_table(self):
        """Test that scalar metrics like crashes generate both canvas and table elements in HTML."""
        # Create ReportGenerator with test data
        generator = ReportGenerator(self.test_data)

        # Generate the HTML report
        html_content = generator.createHTMLReport()

        # Parse the HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")

        # Check that scalar metric sections exist
        windows_crashes_section = soup.find("div", {"id": "Windows-total_crashes"})
        mac_crashes_section = soup.find("div", {"id": "Mac-total_crashes"})

        self.assertIsNotNone(
            windows_crashes_section, "Windows crash scalar section should exist"
        )
        self.assertIsNotNone(
            mac_crashes_section, "Mac crash scalar section should exist"
        )

        # Check that canvas elements exist for crash metrics
        windows_crash_canvas = soup.find(
            "canvas", {"id": "Windows-total_crashes-scalar"}
        )
        mac_crash_canvas = soup.find("canvas", {"id": "Mac-total_crashes-scalar"})

        self.assertIsNotNone(windows_crash_canvas, "Windows crash canvas should exist")
        self.assertIsNotNone(mac_crash_canvas, "Mac crash canvas should exist")

        # Verify canvas has correct attributes
        self.assertEqual(windows_crash_canvas.get("height"), "250px")
        self.assertEqual(mac_crash_canvas.get("height"), "250px")

        # Check that the canvas elements are within their respective scalar sections
        windows_canvas_in_section = windows_crashes_section.find(
            "canvas", {"id": "Windows-total_crashes-scalar"}
        )
        mac_canvas_in_section = mac_crashes_section.find(
            "canvas", {"id": "Mac-total_crashes-scalar"}
        )

        self.assertIsNotNone(
            windows_canvas_in_section, "Windows canvas should be in scalar section"
        )
        self.assertIsNotNone(
            mac_canvas_in_section, "Mac canvas should be in scalar section"
        )

    def test_scalar_metrics_appear_in_summary(self):
        """Test that scalar metrics appear correctly in the summary section with proper links."""
        generator = ReportGenerator(self.test_data)
        html_content = generator.createHTMLReport()
        soup = BeautifulSoup(html_content, "html.parser")

        # Check that crash metrics appear in summary with correct links
        crash_links = soup.find_all("a", href=lambda x: x and "total_crashes" in x)

        self.assertGreater(
            len(crash_links), 0, "Should have crash metric links in summary"
        )

        # Verify specific links exist
        windows_crash_link = soup.find("a", {"href": "#Windows-total_crashes"})
        mac_crash_link = soup.find("a", {"href": "#Mac-total_crashes"})

        self.assertIsNotNone(
            windows_crash_link, "Windows crash summary link should exist"
        )
        self.assertIsNotNone(mac_crash_link, "Mac crash summary link should exist")

        # Verify link text
        self.assertEqual(windows_crash_link.get_text().strip(), "total_crashes")
        self.assertEqual(mac_crash_link.get_text().strip(), "total_crashes")

    def test_scalar_metrics_data_in_charts(self):
        """Test that scalar metrics contain actual data in the chart JavaScript."""
        generator = ReportGenerator(self.test_data)
        html_content = generator.createHTMLReport()

        # Check that the HTML contains the expected crash count values
        self.assertIn(
            "342",
            html_content,
            "Windows control crash count (342) should appear in HTML",
        )
        self.assertIn(
            "292",
            html_content,
            "Windows treatment crash count (292) should appear in HTML",
        )
        self.assertIn(
            "19", html_content, "Mac control crash count (19) should appear in HTML"
        )
        self.assertIn(
            "38", html_content, "Mac treatment crash count (38) should appear in HTML"
        )

        # Check that chart labels exist
        self.assertIn(
            '["control","treatment"]', html_content, "Chart should have branch labels"
        )

        # Check that relative change data appears
        self.assertIn(
            "-14.6", html_content, "Windows relative change (-14.6%) should appear"
        )
        self.assertIn("100.0", html_content, "Mac relative change (100%) should appear")

    def test_mixed_metric_types_in_report(self):
        """Test that reports with both numerical and scalar metrics work correctly."""
        generator = ReportGenerator(self.test_data)
        html_content = generator.createHTMLReport()
        soup = BeautifulSoup(html_content, "html.parser")

        # Check numerical metrics exist
        numerical_sections = soup.find_all(
            "div", {"id": lambda x: x and "fcp_time" in x and "canvas" not in str(x)}
        )
        self.assertGreater(
            len(numerical_sections), 0, "Should have numerical metric sections"
        )

        # Check scalar metrics exist
        scalar_sections = soup.find_all(
            "div",
            {"id": lambda x: x and "total_crashes" in x and "canvas" not in str(x)},
        )
        self.assertGreater(
            len(scalar_sections), 0, "Should have scalar metric sections"
        )

        # Verify both types have titles
        for section in numerical_sections:
            title_div = section.find("div", class_="title")
            self.assertIsNotNone(title_div, "Numerical sections should have titles")

        for section in scalar_sections:
            title_div = section.find("div", class_="title")
            self.assertIsNotNone(title_div, "Scalar sections should have titles")

    def test_report_without_scalar_metrics(self):
        """Test that reports without scalar metrics work correctly (negative test)."""
        # Create test data without scalar metrics
        test_data_no_scalars = self.test_data.copy()
        for branch in ["control", "treatment"]:
            for segment in ["Windows", "Mac"]:
                test_data_no_scalars[branch][segment]["scalar"] = {}

        generator = ReportGenerator(test_data_no_scalars)
        html_content = generator.createHTMLReport()
        soup = BeautifulSoup(html_content, "html.parser")

        # Verify no scalar sections exist
        scalar_sections = soup.find_all(
            "div",
            {"id": lambda x: x and "total_crashes" in x and "canvas" not in str(x)},
        )
        self.assertEqual(
            len(scalar_sections), 0, "Should have no scalar metric sections"
        )

        # Verify no crash canvas elements exist
        crash_canvas = soup.find_all(
            "canvas", {"id": lambda x: x and "total_crashes-scalar" in x}
        )
        self.assertEqual(len(crash_canvas), 0, "Should have no crash canvas elements")

        # But numerical metrics should still exist
        numerical_sections = soup.find_all(
            "div", {"id": lambda x: x and "fcp_time" in x and "canvas" not in str(x)}
        )
        self.assertGreater(
            len(numerical_sections), 0, "Should still have numerical metric sections"
        )


if __name__ == "__main__":
    unittest.main()
