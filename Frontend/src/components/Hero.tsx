import { ArrowRight, Play } from "lucide-react";

const Hero = () => {
  return (
    <section className="relative pt-24 pb-16 md:pt-32 md:pb-20 bg-white overflow-hidden">
      {/* Architectural Grid System */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute left-[10%] top-0 bottom-0 w-px bg-gray-100" />
        <div className="absolute right-[10%] top-0 bottom-0 w-px bg-gray-100" />
        <div className="absolute left-0 right-0 top-[20%] h-px bg-gray-50" />
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="grid lg:grid-cols-2 gap-16 items-center">

          {/* Left Column: Content */}
          <div className="space-y-8">

            {/* Status Indicator */}
            <div className="inline-flex items-center gap-3 px-1 py-1 border-l-2 border-primary pl-4">
              <div className="w-1.5 h-1.5 rounded-full bg-primary" />
              <span className="text-xs font-mono uppercase tracking-[0.2em] text-gray-600">
                AI-Native Security
              </span>
            </div>

            {/* Headline */}
            <div className="space-y-4">
              <h1 className="font-display text-5xl md:text-7xl font-bold tracking-[-0.04em] leading-[0.95] text-gray-900">
                Security at
                <br />
                <span className="text-primary">AI speed.</span>
              </h1>

              {/* Subhead - Simplified */}
              <div className="border-l-2 border-gray-200 pl-6 py-2">
                <p className="text-lg text-gray-600 leading-relaxed max-w-md">
                  Automatically find, fix, and verify security issues in your code.
                  Built for teams shipping AI-generated code at scale.
                </p>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-4">
              <button className="group h-12 px-7 bg-primary text-white font-medium text-sm tracking-wide uppercase overflow-hidden transition-all hover:bg-primary/90 active:scale-[0.98]">
                <span className="flex items-center gap-2">
                  Start Free Scan
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </span>
              </button>

              <button className="h-12 px-7 border-2 border-gray-900 text-gray-900 font-medium text-sm tracking-wide uppercase hover:bg-gray-900 hover:text-white transition-all flex items-center gap-2">
                <Play className="w-3.5 h-3.5 fill-current" />
                Watch Demo
              </button>
            </div>

            {/* Trust Marks */}
            <div className="flex items-center gap-6 pt-4 text-xs font-mono text-gray-500">
              <span>SOC2</span>
              <span className="w-px h-3 bg-gray-200" />
              <span>GDPR</span>
              <span className="w-px h-3 bg-gray-200" />
              <span>Zero Retention</span>
            </div>
          </div>

          {/* Right Column: Visual */}
          <div className="relative">
            <div className="relative aspect-square max-w-md mx-auto">

              {/* Frame */}
              <div className="absolute inset-0 border-2 border-gray-900">

                {/* Corner Accents */}
                <div className="absolute -top-px -left-px w-10 h-10 border-t-4 border-l-4 border-primary" />
                <div className="absolute -top-px -right-px w-10 h-10 border-t-4 border-r-4 border-primary" />
                <div className="absolute -bottom-px -left-px w-10 h-10 border-b-4 border-l-4 border-primary" />
                <div className="absolute -bottom-px -right-px w-10 h-10 border-b-4 border-r-4 border-primary" />

                {/* Content */}
                <div className="absolute inset-3 border border-gray-200 bg-gray-50">
                  <img
                    src="/hero-shield.png"
                    alt="Sentinel Security Platform"
                    className="w-full h-full object-cover"
                  />
                </div>
              </div>

              {/* Floating Stats */}
              <div className="absolute -right-6 top-1/4 bg-white border-2 border-gray-900 px-5 py-3">
                <div className="flex items-baseline gap-2">
                  <div className="w-2 h-2 bg-primary" />
                  <div>
                    <div className="text-xs font-mono text-gray-900 font-bold">LIVE</div>
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider">24/7</div>
                  </div>
                </div>
              </div>

              <div className="absolute -left-6 bottom-1/4 bg-white border-2 border-gray-900 px-5 py-3">
                <div className="flex items-baseline gap-2">
                  <div className="text-xl font-bold font-mono text-gray-900">98%</div>
                  <div className="text-[10px] text-gray-500 uppercase tracking-wider">Accurate</div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
};

export default Hero;