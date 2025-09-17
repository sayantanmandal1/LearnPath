import { Hono } from "npm:hono";
import { cors } from "npm:hono/cors";
import { logger } from "npm:hono/logger";
import { createClient } from "npm:@supabase/supabase-js";
import * as kv from "./kv_store.tsx";

const app = new Hono();

// Initialize Supabase client
const supabase = createClient(
  Deno.env.get('SUPABASE_URL')!,
  Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!,
);

// Enable logger
app.use('*', logger(console.log));

// Enable CORS for all routes and methods
app.use(
  "/*",
  cors({
    origin: "*",
    allowHeaders: ["Content-Type", "Authorization"],
    allowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    exposeHeaders: ["Content-Length"],
    maxAge: 600,
  }),
);

// Health check endpoint
app.get("/make-server-b1dde422/health", (c) => {
  return c.json({ status: "ok" });
});

// User signup endpoint
app.post("/make-server-b1dde422/signup", async (c) => {
  try {
    const { email, password, name } = await c.req.json();
    
    const { data, error } = await supabase.auth.admin.createUser({
      email,
      password,
      user_metadata: { name },
      // Automatically confirm the user's email since an email server hasn't been configured.
      email_confirm: true
    });
    
    if (error) {
      console.log('Signup error:', error);
      return c.json({ error: error.message }, 400);
    }
    
    return c.json({ user: data.user });
  } catch (error) {
    console.log('Server error during signup:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

// Get user profile endpoint
app.get("/make-server-b1dde422/profile", async (c) => {
  try {
    const accessToken = c.req.header('Authorization')?.split(' ')[1];
    
    if (!accessToken) {
      return c.json({ error: 'Authorization header required' }, 401);
    }
    
    const { data: { user }, error } = await supabase.auth.getUser(accessToken);
    
    if (error || !user) {
      return c.json({ error: 'Invalid token' }, 401);
    }
    
    return c.json({ user });
  } catch (error) {
    console.log('Error getting user profile:', error);
    return c.json({ error: 'Internal server error' }, 500);
  }
});

Deno.serve(app.fetch);