"use client";

import dynamic from "next/dynamic";

const ProfilePage = dynamic(() => import("./ProfileClient"), { ssr: false });

export default function Page() {
  return <ProfilePage />;
}
