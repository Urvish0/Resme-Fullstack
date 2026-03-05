-- ============================================================
-- ResMe Database Schema (Fresh Install)
-- Copy-paste into Supabase SQL Editor and hit Run
-- ============================================================

-- Clean slate
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
DROP FUNCTION IF EXISTS public.handle_new_user();
DROP TABLE IF EXISTS public.resumes CASCADE;
DROP TABLE IF EXISTS public.profiles CASCADE;


-- ============================================================
-- 1. PROFILES
-- ============================================================
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT,
  full_name TEXT,
  avatar_url TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"       ON public.profiles FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own profile"     ON public.profiles FOR UPDATE USING (auth.uid() = id);
CREATE POLICY "Service role bypass on profiles"  ON public.profiles FOR ALL USING (true) WITH CHECK (true);

-- Auto-create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name, avatar_url)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data ->> 'full_name', NEW.raw_user_meta_data ->> 'name'),
    NEW.raw_user_meta_data ->> 'avatar_url'
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();


-- ============================================================
-- 2. RESUMES
-- ============================================================
CREATE TABLE public.resumes (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id TEXT NOT NULL,
  content TEXT NOT NULL,
  ats_score INT,
  keywords JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE public.resumes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own resumes"       ON public.resumes FOR SELECT USING (auth.uid()::text = user_id);
CREATE POLICY "Users can insert own resumes"     ON public.resumes FOR INSERT WITH CHECK (auth.uid()::text = user_id);
CREATE POLICY "Service role bypass on resumes"   ON public.resumes FOR ALL USING (true) WITH CHECK (true);

CREATE INDEX idx_resumes_user_id    ON public.resumes(user_id);
CREATE INDEX idx_resumes_created_at ON public.resumes(created_at DESC);


-- ✅ Done! Sign in with Google/GitHub and your profile is auto-created.
