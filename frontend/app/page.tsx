"use client";

import { motion } from "framer-motion";
import {
  ArrowRight,
  Sparkles,
  Shield,
  Zap,
  FileText,
  BarChart3,
  Mail,
} from "lucide-react";
import Link from "next/link";
import AuthButton from "@/components/AuthButton";

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.15,
      duration: 0.6,
      ease: [0.22, 1, 0.36, 1] as [number, number, number, number],
    },
  }),
};

const features = [
  {
    icon: FileText,
    title: "ATS-Optimized Resumes",
    desc: "Multi-agent AI rewrites your resume to pass automated screening with precision keyword alignment.",
  },
  {
    icon: Mail,
    title: "Cover Letters & Outreach",
    desc: "Generate tailored cover letters and cold emails that match the exact tone of the target company.",
  },
  {
    icon: BarChart3,
    title: "Live ATS Scoring",
    desc: "See your before & after ATS compatibility score in real-time as the AI optimizes your content.",
  },
  {
    icon: Shield,
    title: "Human-in-the-Loop",
    desc: "Review the AI's analysis report and provide guidance before final optimization — you stay in control.",
  },
  {
    icon: Zap,
    title: "Multi-Format Export",
    desc: "Download your polished resume as a beautifully formatted PDF with multiple template options.",
  },
  {
    icon: Sparkles,
    title: "Session Memory",
    desc: "The system learns from your past optimizations to deliver increasingly targeted results over time.",
  },
];

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-black text-white selection:bg-white selection:text-black overflow-hidden">
      {/* Auth */}
      <div className="fixed top-6 right-6 z-50">
        <AuthButton />
      </div>

      {/* Decorative gradients */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] bg-purple-600/[0.07] rounded-full blur-[150px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[600px] h-[600px] bg-blue-600/[0.07] rounded-full blur-[150px]" />
      </div>

      {/* Hero */}
      <section className="relative min-h-screen flex flex-col justify-center px-6 lg:px-16 max-w-7xl mx-auto">
        <motion.div
          initial="hidden"
          animate="visible"
          className="space-y-8 max-w-3xl"
        >
          <motion.p
            custom={0}
            variants={fadeUp}
            className="text-xs font-bold tracking-[0.4em] text-white/30 uppercase"
          >
            Precision Career Engineering — v2.0
          </motion.p>

          <motion.h1
            custom={1}
            variants={fadeUp}
            className="text-6xl sm:text-7xl lg:text-8xl font-serif leading-[0.9] tracking-[-0.04em]"
          >
            Your resume,
            <br />
            <span className="italic text-white/50">aligned.</span>
          </motion.h1>

          <motion.p
            custom={2}
            variants={fadeUp}
            className="text-lg sm:text-xl text-white/50 leading-relaxed max-w-xl font-medium"
          >
            ResMe uses a multi-agent AI pipeline to rewrite, score, and optimize
            your resume for any job description — in under 60 seconds.
          </motion.p>

          <motion.div
            custom={3}
            variants={fadeUp}
            className="flex flex-wrap gap-4 pt-4"
          >
            <Link
              href="/tool"
              className="group px-8 py-4 bg-white text-black font-semibold text-sm uppercase tracking-widest flex items-center gap-3 hover:pr-12 transition-all"
            >
              Launch Tool
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/login"
              className="px-8 py-4 border border-white/15 text-sm font-semibold uppercase tracking-widest hover:bg-white/5 transition-colors"
            >
              Sign In
            </Link>
          </motion.div>
        </motion.div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
        >
          <span className="text-[10px] font-bold tracking-[0.3em] text-white/20 uppercase">
            Scroll
          </span>
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-px h-8 bg-gradient-to-b from-white/30 to-transparent"
          />
        </motion.div>
      </section>

      {/* Features Grid */}
      <section className="relative px-6 lg:px-16 py-32 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mb-20"
        >
          <p className="text-xs font-bold tracking-[0.3em] text-white/30 uppercase mb-4">
            Capabilities
          </p>
          <h2 className="text-4xl sm:text-5xl font-serif tracking-tight">
            Built for <span className="italic text-white/50">precision</span>
          </h2>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-px bg-white/[0.06]">
          {features.map((f, i) => (
            <motion.div
              key={f.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="bg-black p-8 lg:p-10 space-y-4 group hover:bg-white/[0.02] transition-colors"
            >
              <f.icon className="w-5 h-5 text-white/30 group-hover:text-white/60 transition-colors" />
              <h3 className="text-base font-semibold tracking-tight">
                {f.title}
              </h3>
              <p className="text-sm text-white/40 leading-relaxed">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section className="relative px-6 lg:px-16 py-32 max-w-7xl mx-auto border-t border-white/[0.06]">
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="mb-20"
        >
          <p className="text-xs font-bold tracking-[0.3em] text-white/30 uppercase mb-4">
            Process
          </p>
          <h2 className="text-4xl sm:text-5xl font-serif tracking-tight">
            Three steps to{" "}
            <span className="italic text-white/50">alignment</span>
          </h2>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-16 lg:gap-8">
          {[
            {
              step: "01",
              title: "Input",
              desc: "Paste your resume and the target job description. Upload a PDF or pull a JD directly from a URL.",
            },
            {
              step: "02",
              title: "Analyze & Review",
              desc: "Our multi-agent pipeline analyzes gaps, extracts keywords, and presents a detailed report for your review.",
            },
            {
              step: "03",
              title: "Optimize & Export",
              desc: "Receive an ATS-optimized resume, cover letter, and cold outreach email — download as PDF instantly.",
            },
          ].map((s, i) => (
            <motion.div
              key={s.step}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.15 }}
              className="space-y-4"
            >
              <span className="text-6xl font-serif text-white/[0.06]">
                {s.step}
              </span>
              <h3 className="text-xl font-semibold tracking-tight">
                {s.title}
              </h3>
              <p className="text-sm text-white/40 leading-relaxed">{s.desc}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* CTA  */}
      <section className="relative px-6 lg:px-16 py-32 max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          className="border border-white/10 p-12 lg:p-20 text-center space-y-8"
        >
          <h2 className="text-4xl sm:text-5xl font-serif tracking-tight">
            Ready to <span className="italic text-white/50">align</span>?
          </h2>
          <p className="text-white/40 max-w-md mx-auto">
            Stop guessing what ATS systems want. Let multi-agent AI do the heavy
            lifting in under a minute.
          </p>
          <Link
            href="/tool"
            className="inline-flex items-center gap-3 px-10 py-5 bg-white text-black font-semibold text-sm uppercase tracking-widest hover:pr-14 transition-all group"
          >
            Get Started
            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
          </Link>
        </motion.div>
      </section>

      {/* Footer */}
      <footer className="px-6 lg:px-16 py-12 max-w-7xl mx-auto border-t border-white/[0.06]">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-white/20 tracking-widest uppercase font-bold">
            ResMe — Precision Career Engineering
          </p>
          <p className="text-xs text-white/20">
            Built with multi-agent AI. Not magic — engineering.
          </p>
        </div>
      </footer>
    </main>
  );
}
