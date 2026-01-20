import { GitMerge, Clock, Zap, Check, Shield } from "lucide-react";
import { AWS, GitHub, Slack, Docker } from "./BrandIcons"; // Import custom brand icons

const AutomationSection = () => {
  return (
    <section className="py-32 bg-white text-foreground">
      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-20 items-center">

          {/* Content */}
          <div className="order-2 lg:order-1">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 border border-blue-100 text-xs font-bold text-blue-600 mb-6 uppercase tracking-wide">
              <Zap className="w-3.5 h-3.5" />
              Zero-Blocking Security
            </div>
            <h2 className="font-display text-4xl md:text-5xl font-bold mb-6 text-foreground leading-tight">
              Security That Runs <br /> Without Blocking You
            </h2>
            <div className="space-y-8 mb-10">
              <p className="text-xl text-muted-foreground leading-relaxed">
                Sentinel activates on code pushes, after AI outputs, and on schedules. No manual steps. No forgotten scans.
              </p>
              <div className="p-6 bg-gray-50 rounded-2xl border border-gray-100">
                <h3 className="font-bold text-lg mb-2 flex items-center gap-2">
                  <Clock className="w-5 h-5 text-blue-600" />
                  Async by Design
                </h3>
                <p className="text-muted-foreground text-sm">
                  Large repos don't freeze workflows. Scans run in the background. Fixes are prepared asynchronously.
                  <span className="block mt-2 font-medium text-foreground italic">"You keep shipping. Sentinel keeps protecting."</span>
                </p>
              </div>
            </div>
          </div>

          {/* Visual: Hub & Spoke Integration Model */}
          <div className="order-1 lg:order-2 relative h-[500px] flex items-center justify-center">
            {/* Background Concentric Rings */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-[450px] h-[450px] border border-dashed border-gray-200 rounded-full opacity-50 animate-spin-slower" />
              <div className="absolute w-[300px] h-[300px] border border-gray-100 rounded-full" />
            </div>

            {/* Center Node (Sentinel) */}
            <div className="relative z-20 w-32 h-32 bg-white rounded-full border-4 border-blue-50 shadow-2xl flex flex-col items-center justify-center text-center">
              <Shield className="w-10 h-10 text-blue-600 mb-2" />
              <span className="font-bold text-xs tracking-wider">SENTINEL</span>
              <span className="text-[10px] text-primary font-mono mt-1">● Active</span>
            </div>

            {/* Orbiting Satellite Nodes */}
            {/* GitHub */}
            <div className="absolute top-10 left-1/4 bg-white p-3 rounded-xl border border-gray-100 shadow-lg flex items-center gap-2 animate-float">
              <GitHub className="w-6 h-6" />
              <div className="text-xs font-bold">GitHub</div>
            </div>

            {/* AWS */}
            <div className="absolute bottom-20 right-10 bg-white p-3 rounded-xl border border-gray-100 shadow-lg flex items-center gap-2 animate-float-delayed">
              <AWS className="w-6 h-6 text-[#FF9900]" />
              <div className="text-xs font-bold">AWS</div>
            </div>

            {/* Slack */}
            <div className="absolute top-1/2 right-0 translate-x-1/2 bg-white p-3 rounded-xl border border-gray-100 shadow-lg flex items-center gap-2 animate-float">
              <Slack className="w-6 h-6" />
              <div className="text-xs font-bold">Slack</div>
            </div>

            {/* Docker */}
            <div className="absolute bottom-10 left-10 bg-white p-3 rounded-xl border border-gray-100 shadow-lg flex items-center gap-2 animate-float-delayed">
              <Docker className="w-6 h-6 text-[#2496ED]" />
              <div className="text-xs font-bold">Docker</div>
            </div>

            {/* Connection Lines (SVG Overlay) */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none -z-10 text-gray-200">
              <line x1="50%" y1="50%" x2="25%" y2="20%" stroke="currentColor" strokeWidth="2" strokeDasharray="5,5" />
              <line x1="50%" y1="50%" x2="80%" y2="80%" stroke="currentColor" strokeWidth="2" strokeDasharray="5,5" />
              <line x1="50%" y1="50%" x2="90%" y2="50%" stroke="currentColor" strokeWidth="2" strokeDasharray="5,5" />
              <line x1="50%" y1="50%" x2="20%" y2="80%" stroke="currentColor" strokeWidth="2" strokeDasharray="5,5" />
            </svg>
          </div>

        </div>
      </div>
    </section>
  );
};

export default AutomationSection;