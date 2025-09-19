import React, { useState } from 'react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export function AuthTest() {
  const [result, setResult] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const testLogin = async () => {
    setLoading(true);
    setResult('Testing login...');
    
    try {
      // Clear any existing tokens
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      
      const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email: 'test@example.com',
          password: 'TestPassword123!'
        }),
      });

      if (response.ok) {
        const data = await response.json();
        localStorage.setItem('access_token', data.access_token);
        localStorage.setItem('refresh_token', data.refresh_token);
        setResult(`✅ Login successful! User: ${data.user.email}`);
        
        // Reload the page to trigger auth context update
        setTimeout(() => {
          window.location.reload();
        }, 1000);
      } else {
        const errorData = await response.json();
        setResult(`❌ Login failed: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error) {
      setResult(`❌ Network error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const clearAuth = () => {
    localStorage.clear();
    sessionStorage.clear();
    setResult('🧹 All auth data cleared');
    setTimeout(() => {
      window.location.reload();
    }, 500);
  };

  return (
    <div className="fixed top-4 right-4 bg-white p-4 border rounded-lg shadow-lg z-50 max-w-sm">
      <h3 className="font-bold mb-2">Auth Test Panel</h3>
      <div className="space-y-2">
        <button
          onClick={testLogin}
          disabled={loading}
          className="w-full px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? 'Testing...' : 'Test Login'}
        </button>
        <button
          onClick={clearAuth}
          className="w-full px-3 py-2 bg-red-500 text-white rounded hover:bg-red-600"
        >
          Clear All Auth Data
        </button>
        {result && (
          <div className="text-sm p-2 bg-gray-100 rounded">
            {result}
          </div>
        )}
      </div>
      <div className="text-xs text-gray-500 mt-2">
        Test credentials:<br/>
        Email: test@example.com<br/>
        Password: TestPassword123!
      </div>
    </div>
  );
}