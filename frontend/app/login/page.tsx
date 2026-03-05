"use client";

import dynamic from "next/dynamic";

// Prevent Next.js from pre-rendering this page at build time.
// The Supabase client requires browser-only env vars that aren't available during static generation.
const LoginPage = dynamic(() => import("./LoginClient"), { ssr: false });

export default function Page() {
  return <LoginPage />;
}
