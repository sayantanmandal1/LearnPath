
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};


serve(async (req) => {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  console.log('delete_user function called');
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  let user_id = null;
  try {
    const body = await req.json();
    user_id = body.user_id;
    console.log('user_id received:', user_id);
  } catch (e) {
    console.log('Error parsing request body:', e);
    return new Response(JSON.stringify({ error: "Invalid request body" }), { status: 400, headers: corsHeaders });
  }

  if (!user_id) {
    console.log('Missing user_id');
    return new Response(JSON.stringify({ error: "Missing user_id" }), { status: 400, headers: corsHeaders });
  }

  // Delete user data from all tables, check for errors
  const tables = [
    { table: "profiles", column: "id" },
    { table: "skills", column: "user_id" },
    { table: "achievements", column: "user_id" },
    { table: "activities", column: "user_id" }
  ];

  for (const { table, column } of tables) {
    const { error } = await supabase.from(table).delete().eq(column, user_id);
    if (error) {
      console.log(`Failed to delete from ${table}:`, error.message);
      return new Response(JSON.stringify({ error: `Failed to delete from ${table}: ${error.message}` }), { status: 500, headers: corsHeaders });
    } else {
      console.log(`Deleted from ${table} where ${column} = ${user_id}`);
    }
  }

  // Delete the user from Auth
  const { error: authError } = await supabase.auth.admin.deleteUser(user_id);

  if (authError) {
    console.log('Auth delete error:', authError.message);
    return new Response(JSON.stringify({ error: `Auth delete error: ${authError.message}` }), { status: 500, headers: corsHeaders });
  }

  console.log('User deleted successfully:', user_id);
  return new Response(JSON.stringify({ success: true }), { status: 200, headers: corsHeaders });
});
