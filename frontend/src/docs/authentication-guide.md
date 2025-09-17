# CareerPilot Authentication Guide

## How Authentication Works

CareerPilot uses Supabase Auth for secure user authentication and session management. Here's how the authentication flow works:

### 1. Sign In Options

Users can sign in using:
- **Email & Password**: Traditional email/password authentication
- **GitHub OAuth**: Social login with GitHub
- **LinkedIn OAuth**: Social login with LinkedIn  
- **Google OAuth**: Social login with Google (configurable)

### 2. Sign Out Process

To sign out of an account, users can:

#### Option 1: Using the Navigation Menu
1. Click on the user avatar in the top-right corner
2. Select "Log out" from the dropdown menu
3. User will be automatically signed out and redirected

#### Option 2: Programmatic Sign Out
```tsx
import { authUtils } from '../utils/auth';

const handleSignOut = async () => {
  const result = await authUtils.signOut();
  if (result.success) {
    // User successfully signed out
    console.log('User signed out');
  }
};
```

### 3. Authentication State Management

The app manages authentication state in `App.tsx`:

```tsx
const [user, setUser] = useState<User | null>(null);
const [loading, setLoading] = useState(true);

useEffect(() => {
  // Get initial session
  supabase.auth.getSession().then(({ data: { session } }) => {
    setUser(session?.user ?? null);
    setLoading(false);
  });

  // Listen for auth changes
  const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
    setUser(session?.user ?? null);
    setLoading(false);
  });

  return () => subscription.unsubscribe();
}, []);
```

### 4. Auth Utilities

The `authUtils` module provides convenient methods:

```tsx
// Sign out
await authUtils.signOut();

// Get current user
const { user } = await authUtils.getUser();

// Sign in with email/password
const result = await authUtils.signInWithPassword(email, password);

// Sign up
const result = await authUtils.signUp(email, password, { name: 'John Doe' });

// OAuth sign in
const result = await authUtils.signInWithOAuth('github');
```

### 5. Protected Routes

Routes like Dashboard, Profile, and Analysis are protected:

```tsx
<Route path="/dashboard" element={
  isLoggedIn ? <Dashboard /> : <LandingPage onLogin={() => setShowLoginModal(true)} />
} />
```

### 6. User Data Access

Once authenticated, user data is available throughout the app:

```tsx
// In Dashboard component
const [user, setUser] = useState<any>(null);

useEffect(() => {
  const getUser = async () => {
    const { data: { user } } = await supabase.auth.getUser();
    setUser(user);
  };
  
  getUser();
}, []);

// Display user info
<h1>Welcome back, {user?.user_metadata?.name || user?.email?.split('@')[0]}!</h1>
```

### 7. Session Persistence

Supabase automatically handles session persistence:
- Sessions are stored securely in localStorage
- Users remain logged in across browser sessions
- Sessions automatically refresh when needed
- Auth state is synchronized across tabs

### 8. Error Handling

All authentication actions include comprehensive error handling:

```tsx
try {
  const { error } = await supabase.auth.signOut();
  if (error) throw error;
  toast.success('Signed out successfully');
} catch (error: any) {
  toast.error('Error signing out: ' + error.message);
}
```

## Complete Sign Out Flow

1. **User Clicks Sign Out**: Either from dropdown menu or programmatically
2. **Auth Utils Called**: `authUtils.signOut()` is invoked
3. **Supabase Sign Out**: `supabase.auth.signOut()` clears the session
4. **State Updated**: Auth state listener updates `user` to `null`
5. **UI Updates**: Navigation shows "Sign In" button, protected routes redirect
6. **Toast Notification**: User sees "Signed out successfully" message

## OAuth Setup Requirements

For social login to work, you need to configure OAuth providers in Supabase:

### GitHub OAuth
1. Go to Supabase Dashboard → Authentication → Providers
2. Enable GitHub provider
3. Add GitHub OAuth app credentials
4. Set redirect URL: `https://your-project.supabase.co/auth/v1/callback`

### LinkedIn OAuth  
1. Enable LinkedIn provider in Supabase
2. Create LinkedIn OAuth app
3. Configure redirect URLs

⚠️ **Important**: OAuth providers must be configured in Supabase dashboard for social login to work properly.

## Security Features

- **Secure Sessions**: JWT tokens with automatic refresh
- **Protected Routes**: Server-side session validation
- **CSRF Protection**: Built-in CSRF protection
- **Rate Limiting**: Login attempt rate limiting
- **Password Security**: Secure password hashing
- **OAuth Security**: Secure OAuth flow implementation

## Troubleshooting

### Common Issues:

1. **OAuth Not Working**: Check provider configuration in Supabase dashboard
2. **Session Not Persisting**: Check localStorage and Supabase project settings
3. **Sign Out Not Working**: Check network connectivity and Supabase status
4. **Redirect Issues**: Verify redirect URLs in OAuth provider settings

### Debug Authentication:

```tsx
// Check current session
const { data: { session } } = await supabase.auth.getSession();
console.log('Current session:', session);

// Check auth events
supabase.auth.onAuthStateChange((event, session) => {
  console.log('Auth event:', event, session);
});
```

This comprehensive authentication system provides a secure, user-friendly experience with multiple sign-in options and proper session management.