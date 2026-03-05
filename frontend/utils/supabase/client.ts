import { createBrowserClient } from "@supabase/ssr"

export function createClient() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseAnonKey) {
    // During Docker build / static generation, env vars may not be available.
    // Return a no-op proxy that won't crash but also won't do anything.
    // The real client is only created at runtime in the browser.
    return null as any
  }

  return createBrowserClient(supabaseUrl, supabaseAnonKey)
}
