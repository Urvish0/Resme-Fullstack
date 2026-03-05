import { createBrowserClient } from "@supabase/ssr"

export function createClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  // Diagnostic logging for build-time variables
  if (typeof window !== "undefined") {
    const isUrlSet = !!supabaseUrl && supabaseUrl !== "undefined" && supabaseUrl !== "";
    const isKeySet = !!supabaseAnonKey && supabaseAnonKey !== "undefined" && supabaseAnonKey !== "";
    
    if (!isUrlSet || !isKeySet) {
      console.error(
        `[Supabase Client] Environment variables missing!
         URL present: ${isUrlSet} (Value length: ${supabaseUrl?.length || 0})
         Key present: ${isKeySet} (Value length: ${supabaseAnonKey?.length || 0})
         
         Vercel Fix:
         1. Go to Vercel -> Settings -> Environment Variables.
         2. Ensure keys match EXACTLY: NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY.
         3. Ensure "Environment" checkboxes (Production, Preview, Development) are ALL checked.
         4. Trigger a NEW DEPLOYMENT (Redeploy with "Clear Build Cache").`
      );
    }
  }

  if (!supabaseUrl || supabaseUrl === "undefined" || supabaseUrl === "" || !supabaseAnonKey || supabaseAnonKey === "undefined" || supabaseAnonKey === "") {
    // Return a dummy client to prevent crash
    return {
      auth: {
        getSession: async () => ({ data: { session: null }, error: new Error("Supabase URL missing. Check Vercel Environment Variables.") }),
        onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
        signInWithOAuth: async () => ({ error: new Error("Auth unavailable: Supabase keys missing during build.") }),
        signOut: async () => {},
        getUser: async () => ({ data: { user: null }, error: null }),
      },
    } as any
  }

  return createBrowserClient(supabaseUrl, supabaseAnonKey)
}
