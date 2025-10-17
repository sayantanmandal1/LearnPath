"""
Debug JWT token verification
"""
import base64
import json
from jose import jwt

def decode_jwt_payload(token):
    """Decode JWT payload without verification to see the content"""
    try:
        # Split token into parts
        header, payload, signature = token.split('.')
        
        # Add padding if needed
        payload += '=' * (4 - len(payload) % 4)
        
        # Decode payload
        decoded_payload = base64.urlsafe_b64decode(payload.encode())
        payload_json = json.loads(decoded_payload)
        
        print("JWT Payload:")
        print(json.dumps(payload_json, indent=2))
        return payload_json
    except Exception as e:
        print(f"Error decoding JWT: {e}")
        return None

def test_jwt_verification():
    """Test JWT verification with Supabase secret"""
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzI3NTI3NDMxLCJpYXQiOjE3Mjc1MjM4MzEsImlzcyI6Imh0dHBzOi8vaW10bmhjcGFna2V3cXRlZHRiaWEuc3VwYWJhc2UuY28vYXV0aC92MSIsInN1YiI6IjNhNzdjNGM5LTM4OTQtNDY0Ny04YTA4LWU1ZDhhMWVkZDU5MCIsImVtYWlsIjoibXVrdWxAZ21haWwuY29tIiwicGhvbmUiOiIiLCJhcHBfbWV0YWRhdGEiOnsicHJvdmlkZXIiOiJlbWFpbCIsInByb3ZpZGVycyI6WyJlbWFpbCJdfSwidXNlcl9tZXRhZGF0YSI6eyJjYXJlZXJHb2FsIjoiU29mdHdhcmUgRW5naW5lZXIiLCJleHBlcmllbmNlIjoic3R1ZGVudCIsIm5hbWUiOiJNdWt1bCJ9LCJyb2xlIjoiYXV0aGVudGljYXRlZCIsImFhbCI6ImFhbDEiLCJhbXIiOlt7Im1ldGhvZCI6InBhc3N3b3JkIiwidGltZXN0YW1wIjoxNzI3NTIzODMxfV0sInNlc3Npb25faWQiOiI4NDdmN2Q1YS1jZGEyLTRhNzQtOWUyOC1jODE2MDc1ODRhNzUifQ.WQiQIIzMTNIHjViNiQzMTN5LTQyYTYTYjcwMC0zjh1MWljQWnNmOQil1CJpc19hbm9ueUVvdXM"
    
    supabase_secret = "cruD3Zok1rCj0tfi8/YX01cY5sZEvNT3VsCl0XZLeun9ux93uvWORw4Xl7mqy6o9VNthSjnV0ZlNJ6M/Qe9osA=="
    
    print("Testing JWT verification...")
    
    # First, decode payload to see content
    payload = decode_jwt_payload(token)
    
    if payload:
        print(f"\nUser ID (sub): {payload.get('sub')}")
        print(f"Email: {payload.get('email')}")
        print(f"Audience: {payload.get('aud')}")
        print(f"Issuer: {payload.get('iss')}")
        
        # Test verification with Supabase secret
        try:
            print("\nTrying verification with Supabase secret...")
            options = {"verify_aud": False}  # Disable audience verification
            verified_payload = jwt.decode(
                token,
                supabase_secret,
                algorithms=["HS256"],
                options=options
            )
            print("✅ JWT verification successful!")
            print(f"Subject: {verified_payload.get('sub')}")
        except Exception as e:
            print(f"❌ JWT verification failed: {e}")
            
            # Try without audience verification
            try:
                print("Trying without any verification...")
                verified_payload = jwt.decode(
                    token,
                    supabase_secret,
                    algorithms=["HS256"],
                    options={"verify_signature": True, "verify_aud": False, "verify_iss": False, "verify_exp": False}
                )
                print("✅ JWT verification successful with relaxed options!")
            except Exception as e2:
                print(f"❌ Still failed: {e2}")

if __name__ == "__main__":
    test_jwt_verification()