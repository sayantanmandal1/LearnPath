// Centralized API utility for backend calls
// Update BASE_URL as needed for deployment

const BASE_URL = "http://127.0.0.1:8000/api/v1";

export async function apiGet(path: string, token?: string) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  const res = await fetch(`${BASE_URL}${path}`, {
    headers,
    // Remove credentials: 'include' since we're using CORS with allow_origins=*
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function apiPost(path: string, data: any, token?: string) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(data),
    // Remove credentials: 'include' since we're using CORS with allow_origins=*
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function apiPut(path: string, data: any, token?: string) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'PUT',
    headers,
    body: JSON.stringify(data),
    // Remove credentials: 'include' since we're using CORS with allow_origins=*
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function apiDelete(path: string, token?: string) {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json'
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'DELETE',
    headers,
    // Remove credentials: 'include' since we're using CORS with allow_origins=*
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
