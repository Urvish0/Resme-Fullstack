import { createBrowserClient } from "@supabase/ssr"

export function createClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  // Diagnostic logging for build-time variables
  if (typeof window !== "undefined") {
    const isUrlSet = !!supabaseUrl && supabaseUrl !== "undefined";
    const isKeySet = !!supabaseAnonKey && supabaseAnonKey !== "undefined";
    
    if (!isUrlSet || !isKeySet) {
      console.error(
        `[Supabase Client] Environment variables missing!
         URL present: ${isUrlSet} 
         Key present: ${isKeySet}
         Value of URL (masked): ${supabaseUrl ? supabaseUrl.substring(0, 10) + "..." : "missing"}
         
         IMPORTANT: If you see "false" or "undefined" above, you must add these as 
         DOCKER BUILD ARGS in the Render dashboard (Settings -> Advanced).`
      );
    }
  }

  if (!supabaseUrl || supabaseUrl === "undefined" || !supabaseAnonKey || supabaseAnonKey === "undefined") {
    // Return a dummy client that provides useful errors instead of crashing with TypeError
    return {
      auth: {
        getSession: async () => ({ data: { session: null }, error: new Error("Supabase URL is missing. Check Build Args.") }),
        onAuthStateChange: () => ({ data: { subscription: { unsubscribe: () => {} } } }),
        signInWithOAuth: async () => ({ error: new Error("Auth unavailable: Supabase URL is missing during build.") }),
        signOut: async () => {},
        getUser: async () => ({ data: { user: null }, error: null }),
      },
    } as any
  }

  return createBrowserClient(supabaseUrl, supabaseAnonKey)
}
