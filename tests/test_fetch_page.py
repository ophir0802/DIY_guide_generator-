"""
Unit tests for the fetch_page function from crawler.py
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from crawler import fetch_page


class TestFetchPage:
    """Test cases for the fetch_page function"""

    @patch('crawler.time.sleep')
    @patch('crawler.random.uniform')
    @patch('crawler.random.choice')
    @patch('crawler.requests.get')
    def test_fetch_page_success(self, mock_get, mock_choice, mock_uniform, mock_sleep):
        """Test successful page fetch"""
        # Setup mocks
        mock_choice.return_value = "Mozilla/5.0 (Test Browser)"
        mock_uniform.return_value = 2.5  # Fixed delay for testing
        mock_response = Mock()
        mock_response.text = "<html><body>Test Content</body></html>"
        mock_response.raise_for_status = Mock()  # No exception raised
        mock_get.return_value = mock_response

        # Execute
        result = fetch_page("https://www.example.com/test")

        # Assert
        assert result == "<html><body>Test Content</body></html>"
        mock_sleep.assert_called_once_with(2.5)
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://www.example.com/test"
        assert "User-Agent" in call_args[1]["headers"]
        assert call_args[1]["timeout"] == 10

    @patch('crawler.time.sleep')
    @patch('crawler.random.uniform')
    @patch('crawler.random.choice')
    @patch('crawler.requests.get')
    def test_fetch_page_http_error(self, mock_get, mock_choice, mock_uniform, mock_sleep):
        """Test handling of HTTP error (4xx, 5xx)"""
        # Setup mocks
        mock_choice.return_value = "Mozilla/5.0 (Test Browser)"
        mock_uniform.return_value = 2.5
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_get.return_value = mock_response

        # Execute
        result = fetch_page("https://www.example.com/notfound")

        # Assert
        assert result is None
        mock_get.assert_called_once()

    @patch('crawler.time.sleep')
    @patch('crawler.random.uniform')
    @patch('crawler.random.choice')
    @patch('crawler.requests.get')
    def test_fetch_page_connection_error(self, mock_get, mock_choice, mock_uniform, mock_sleep):
        """Test handling of connection error"""
        # Setup mocks
        mock_choice.return_value = "Mozilla/5.0 (Test Browser)"
        mock_uniform.return_value = 2.5
        mock_get.side_effect = requests.ConnectionError("Connection failed")

        # Execute
        result = fetch_page("https://www.example.com/test")

        # Assert
        assert result is None
        mock_get.assert_called_once()

    @patch('crawler.time.sleep')
    @patch('crawler.random.uniform')
    @patch('crawler.random.choice')
    @patch('crawler.requests.get')
    def test_fetch_page_timeout(self, mock_get, mock_choice, mock_uniform, mock_sleep):
        """Test handling of timeout error"""
        # Setup mocks
        mock_choice.return_value = "Mozilla/5.0 (Test Browser)"
        mock_uniform.return_value = 2.5
        mock_get.side_effect = requests.Timeout("Request timed out")

        # Execute
        result = fetch_page("https://www.example.com/test")

        # Assert
        assert result is None
        mock_get.assert_called_once()

    @patch('crawler.time.sleep')
    @patch('crawler.random.uniform')
    @patch('crawler.random.choice')
    @patch('crawler.requests.get')
    def test_fetch_page_headers_set_correctly(self, mock_get, mock_choice, mock_uniform, mock_sleep):
        """Test that headers are set correctly with random user agent"""
        # Setup mocks
        test_user_agent = "Mozilla/5.0 (Custom Test Agent)"
        mock_choice.return_value = test_user_agent
        mock_uniform.return_value = 2.5
        mock_response = Mock()
        mock_response.text = "Test"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Execute
        fetch_page("https://www.example.com/test")

        # Assert headers
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert headers["User-Agent"] == test_user_agent
        assert "Accept-Language" in headers
        assert headers["Accept-Language"] == "en-US,en;q=0.9"

    @patch('crawler.time.sleep')
    @patch('crawler.random.uniform')
    @patch('crawler.random.choice')
    @patch('crawler.requests.get')
    def test_fetch_page_delay_called(self, mock_get, mock_choice, mock_uniform, mock_sleep):
        """Test that delay is called before making request"""
        # Setup mocks
        mock_choice.return_value = "Mozilla/5.0 (Test Browser)"
        mock_uniform.return_value = 3.7
        mock_response = Mock()
        mock_response.text = "Test"
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Execute
        fetch_page("https://www.example.com/test")

        # Assert delay was called with correct value
        mock_uniform.assert_called_once_with(2, 5)
        mock_sleep.assert_called_once_with(3.7)
        # Verify sleep was called before get
        assert mock_sleep.call_count == 1
        assert mock_get.call_count == 1

    @patch('crawler.time.sleep')
    @patch('crawler.random.uniform')
    @patch('crawler.random.choice')
    @patch('crawler.requests.get')
    def test_fetch_page_empty_response(self, mock_get, mock_choice, mock_uniform, mock_sleep):
        """Test handling of empty response"""
        # Setup mocks
        mock_choice.return_value = "Mozilla/5.0 (Test Browser)"
        mock_uniform.return_value = 2.5
        mock_response = Mock()
        mock_response.text = ""  # Empty response
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        # Execute
        result = fetch_page("https://www.example.com/test")

        # Assert
        assert result == ""
        mock_get.assert_called_once()

    @patch('crawler.time.sleep')
    @patch('crawler.random.uniform')
    @patch('crawler.random.choice')
    @patch('crawler.requests.get')
    def test_fetch_page_request_exception(self, mock_get, mock_choice, mock_uniform, mock_sleep):
        """Test handling of generic RequestException"""
        # Setup mocks
        mock_choice.return_value = "Mozilla/5.0 (Test Browser)"
        mock_uniform.return_value = 2.5
        mock_get.side_effect = requests.RequestException("Generic request error")

        # Execute
        result = fetch_page("https://www.example.com/test")

        # Assert
        assert result is None
        mock_get.assert_called_once()

