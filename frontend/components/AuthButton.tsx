"use client";

import { createClient } from "@/utils/supabase/client";
import { useEffect, useState } from "react";
import { User, AuthChangeEvent, Session } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { LogOut, User as UserIcon } from "lucide-react";

export default function AuthButton() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const supabase = createClient();
  const router = useRouter();

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return;
    }

    const getUser = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();
      setUser(user);
      setLoading(false);
    };

    getUser();

    const { data: authListener } = supabase.auth.onAuthStateChange(
      (event: AuthChangeEvent, session: Session | null) => {
        setUser(session?.user ?? null);
      },
    );

    return () => {
      authListener.subscription.unsubscribe();
    };
  }, [supabase]);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.refresh();
  };

  if (loading) {
    return (
      <div className="w-8 h-8 rounded-full bg-white/10 animate-pulse border border-white/20" />
    );
  }

  if (!user) {
    return (
      <button
        onClick={() => router.push("/login")}
        className="px-4 py-2 border border-white/20 rounded-full text-xs font-bold uppercase tracking-widest hover:bg-white hover:text-black transition-colors"
      >
        Sign In
      </button>
    );
  }

  return (
    <div className="group relative">
      <button className="flex items-center gap-2 p-1 pr-3 border border-white/20 rounded-full hover:bg-white/5 transition-colors">
        <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-blue-500 flex items-center justify-center overflow-hidden">
          {user.user_metadata?.avatar_url ? (
            <img
              src={user.user_metadata.avatar_url}
              alt="Avatar"
              className="w-full h-full object-cover"
            />
          ) : (
            <UserIcon className="w-4 h-4 text-white" />
          )}
        </div>
        <span className="text-xs font-medium max-w-[100px] truncate">
          {user.user_metadata?.full_name || user.email?.split("@")[0]}
        </span>
      </button>

      {/* Dropdown */}
      <div className="absolute right-0 mt-2 w-48 bg-black border border-white/10 rounded-xl shadow-2xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all transform origin-top-right group-hover:translate-y-0 translate-y-2">
        <div className="p-2 space-y-1">
          <div className="px-3 py-2 border-b border-white/10 mb-1">
            <p className="text-[10px] uppercase font-bold text-white/40 tracking-widest mb-1">
              Account
            </p>
            <p className="text-xs text-white/80 truncate">{user.email}</p>
          </div>
          <button
            onClick={() => router.push("/profile")}
            className="w-full text-left px-3 py-2 text-xs font-medium text-white/70 hover:bg-white/5 rounded-lg transition-colors flex items-center justify-between"
          >
            Profile
            <UserIcon className="w-3 h-3" />
          </button>
          <button
            onClick={handleSignOut}
            className="w-full text-left px-3 py-2 text-xs font-medium text-red-400 hover:bg-white/5 rounded-lg transition-colors flex items-center justify-between"
          >
            Sign Out
            <LogOut className="w-3 h-3" />
          </button>
        </div>
      </div>
    </div>
  );
}
