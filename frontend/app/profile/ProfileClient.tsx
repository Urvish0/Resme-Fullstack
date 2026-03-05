"use client";

import { createClient } from "@/utils/supabase/client";
import { useEffect, useState } from "react";
import { User } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import {
  ArrowLeft,
  Mail,
  Calendar,
  FileText,
  Activity,
  LogOut,
  Shield,
  Pencil,
  Check,
  X,
} from "lucide-react";
import { motion } from "framer-motion";
import toast from "react-hot-toast";

interface ResumeEntry {
  id: string;
  ats_score: number | null;
  created_at: string;
}

export default function ProfilePage() {
  const supabase = createClient();
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [resumes, setResumes] = useState<ResumeEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingName, setEditingName] = useState(false);
  const [nameInput, setNameInput] = useState("");
  const [savingName, setSavingName] = useState(false);

  useEffect(() => {
    if (!supabase) {
      setLoading(false);
      return;
    }

    const load = async () => {
      const {
        data: { user },
      } = await supabase.auth.getUser();

      if (!user) {
        router.push("/login");
        return;
      }

      setUser(user);

      // Fetch resume history
      const { data } = await supabase
        .from("resumes")
        .select("id, ats_score, created_at")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false })
        .limit(20);

      setResumes(data ?? []);
      setLoading(false);
    };

    load();
  }, [supabase, router]);

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    toast.success("Signed out successfully");
    router.push("/");
  };

  const handleSaveName = async () => {
    if (!nameInput.trim() || !supabase || !user) return;
    setSavingName(true);
    try {
      // Update Supabase auth user metadata
      await supabase.auth.updateUser({
        data: { full_name: nameInput.trim() },
      });
      // Update the profiles table
      await supabase
        .from("profiles")
        .update({
          full_name: nameInput.trim(),
          updated_at: new Date().toISOString(),
        })
        .eq("id", user.id);

      // Refresh user state
      const {
        data: { user: refreshed },
      } = await supabase.auth.getUser();
      if (refreshed) setUser(refreshed);

      toast.success("Name updated");
      setEditingName(false);
    } catch {
      toast.error("Failed to update name");
    } finally {
      setSavingName(false);
    }
  };

  const avgScore =
    resumes.filter((r) => r.ats_score != null).length > 0
      ? Math.round(
          resumes
            .filter((r) => r.ats_score != null)
            .reduce((a, b) => a + (b.ats_score ?? 0), 0) /
            resumes.filter((r) => r.ats_score != null).length,
        )
      : null;

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-white/20 border-t-white rounded-full animate-spin" />
      </div>
    );
  }

  if (!user) return null;

  const joinDate = new Date(user.created_at).toLocaleDateString("en-US", {
    month: "long",
    year: "numeric",
  });

  return (
    <main className="min-h-screen bg-black text-white">
      {/* Decorative gradients */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] bg-purple-600/[0.05] rounded-full blur-[150px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-blue-600/[0.05] rounded-full blur-[150px]" />
      </div>

      <div className="relative max-w-3xl mx-auto px-6 py-16">
        {/* Back */}
        <motion.button
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={() => router.back()}
          className="flex items-center gap-2 text-xs font-bold tracking-widest text-white/30 uppercase hover:text-white/60 transition-colors mb-12"
        >
          <ArrowLeft className="w-3 h-3" />
          Back
        </motion.button>

        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-6 mb-16"
        >
          <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-purple-500 to-blue-500 flex items-center justify-center overflow-hidden shrink-0">
            {user.user_metadata?.avatar_url ? (
              <img
                src={user.user_metadata.avatar_url}
                alt="Avatar"
                className="w-full h-full object-cover"
              />
            ) : (
              <span className="text-2xl font-bold">
                {(user.email?.[0] ?? "U").toUpperCase()}
              </span>
            )}
          </div>
          <div className="flex-1 min-w-0">
            {editingName ? (
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={nameInput}
                  onChange={(e) => setNameInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSaveName()}
                  autoFocus
                  className="text-2xl font-semibold tracking-tight bg-transparent border-b border-white/20 focus:border-white/60 outline-none w-full py-1 transition-colors"
                />
                <button
                  onClick={handleSaveName}
                  disabled={savingName}
                  className="p-1.5 hover:bg-white/10 rounded-lg transition-colors text-green-400"
                >
                  <Check className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setEditingName(false)}
                  className="p-1.5 hover:bg-white/10 rounded-lg transition-colors text-white/40"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-semibold tracking-tight">
                  {user.user_metadata?.full_name ||
                    user.user_metadata?.name ||
                    user.email?.split("@")[0]}
                </h1>
                <button
                  onClick={() => {
                    setNameInput(
                      user.user_metadata?.full_name ||
                        user.user_metadata?.name ||
                        user.email?.split("@")[0] ||
                        "",
                    );
                    setEditingName(true);
                  }}
                  className="p-1.5 hover:bg-white/10 rounded-lg transition-colors text-white/20 hover:text-white/60"
                >
                  <Pencil className="w-3.5 h-3.5" />
                </button>
              </div>
            )}
            <p className="text-sm text-white/40 mt-1">{user.email}</p>
          </div>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 sm:grid-cols-3 gap-px bg-white/[0.06] mb-16"
        >
          {[
            {
              icon: Calendar,
              label: "Member Since",
              value: joinDate,
            },
            {
              icon: FileText,
              label: "Optimizations",
              value: resumes.length.toString(),
            },
            {
              icon: Activity,
              label: "Avg ATS Score",
              value: avgScore ? `${avgScore}%` : "—",
            },
          ].map((stat) => (
            <div
              key={stat.label}
              className="bg-black p-6 flex items-center gap-4"
            >
              <stat.icon className="w-4 h-4 text-white/20" />
              <div>
                <p className="text-[10px] font-bold tracking-widest text-white/30 uppercase">
                  {stat.label}
                </p>
                <p className="text-lg font-semibold mt-1">{stat.value}</p>
              </div>
            </div>
          ))}
        </motion.div>

        {/* Resume History */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-xs font-bold tracking-[0.3em] text-white/30 uppercase mb-6">
            Optimization History
          </h2>

          {resumes.length === 0 ? (
            <div className="border border-white/[0.06] p-12 text-center">
              <FileText className="w-6 h-6 text-white/10 mx-auto mb-4" />
              <p className="text-sm text-white/30">
                No optimizations yet. Head to the{" "}
                <button
                  onClick={() => router.push("/tool")}
                  className="text-white/60 underline underline-offset-4 hover:text-white transition-colors"
                >
                  tool
                </button>{" "}
                to get started.
              </p>
            </div>
          ) : (
            <div className="space-y-px">
              {resumes.map((resume, i) => (
                <motion.div
                  key={resume.id}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.05 * i }}
                  className="flex items-center justify-between bg-white/[0.02] hover:bg-white/[0.04] transition-colors px-5 py-4 border border-white/[0.04]"
                >
                  <div className="flex items-center gap-4">
                    <FileText className="w-4 h-4 text-white/20" />
                    <div>
                      <p className="text-sm font-medium">
                        Optimization #{resumes.length - i}
                      </p>
                      <p className="text-xs text-white/30 mt-0.5">
                        {new Date(resume.created_at).toLocaleDateString(
                          "en-US",
                          {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          },
                        )}
                      </p>
                    </div>
                  </div>
                  {resume.ats_score != null && (
                    <div
                      className={`text-sm font-bold px-3 py-1 rounded-full ${
                        resume.ats_score >= 80
                          ? "bg-green-500/10 text-green-400"
                          : resume.ats_score >= 60
                            ? "bg-yellow-500/10 text-yellow-400"
                            : "bg-red-500/10 text-red-400"
                      }`}
                    >
                      {resume.ats_score}%
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>

        {/* Account Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-16"
        >
          <h2 className="text-xs font-bold tracking-[0.3em] text-white/30 uppercase mb-6">
            Account
          </h2>

          <div className="space-y-px">
            <div className="flex items-center justify-between bg-white/[0.02] px-5 py-4 border border-white/[0.04]">
              <div className="flex items-center gap-4">
                <Mail className="w-4 h-4 text-white/20" />
                <div>
                  <p className="text-sm font-medium">Email</p>
                  <p className="text-xs text-white/30 mt-0.5">{user.email}</p>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between bg-white/[0.02] px-5 py-4 border border-white/[0.04]">
              <div className="flex items-center gap-4">
                <Shield className="w-4 h-4 text-white/20" />
                <div>
                  <p className="text-sm font-medium">Auth Provider</p>
                  <p className="text-xs text-white/30 mt-0.5 capitalize">
                    {user.app_metadata?.provider ?? "email"}
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={handleSignOut}
              className="w-full flex items-center justify-between bg-white/[0.02] hover:bg-red-500/[0.05] px-5 py-4 border border-white/[0.04] transition-colors group"
            >
              <div className="flex items-center gap-4">
                <LogOut className="w-4 h-4 text-red-400/50 group-hover:text-red-400 transition-colors" />
                <p className="text-sm font-medium text-red-400/70 group-hover:text-red-400 transition-colors">
                  Sign Out
                </p>
              </div>
            </button>
          </div>
        </motion.div>
      </div>
    </main>
  );
}
