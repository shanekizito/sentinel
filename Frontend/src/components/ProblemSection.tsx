import { AlertTriangle, Repeat, Clock, FileWarning } from "lucide-react";

const ProblemSection = () => {
  return (
    <section className="py-24 bg-gray-50 border-y-2 border-gray-900 relative overflow-hidden">
      {/* Structural Grid */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gray-200" />
      </div>

      <div className="container mx-auto px-6">
        <div className="grid lg:grid-cols-2 gap-20 items-start">

          {/* Left: Narrative */}
          <div className="sticky top-32">
            <div className="space-y-8">
              <div className="inline-block border-l-4 border-red-600 pl-4">
                <div className="text-xs font-mono uppercase tracking-[0.2em] text-red-600 mb-2">
                  The Problem
                </div>
              </div>

              <h2 className="font-display text-4xl md:text-5xl font-bold text-gray-900 tracking-tight leading-[1.1]">
                AI writes code fast.
                <br />
                <span className="text-gray-500">Is it secure?</span>
              </h2>

              <p className="text-xl text-gray-600 leading-relaxed border-l-2 border-gray-200 pl-6">
                Developers are moving faster than ever. Traditional security reviews can't keep up with the flood of AI-generated code.
              </p>
            </div>

            {/* Terminal Visual */}
            <div className="mt-12 border-2 border-gray-900 bg-black p-6">
              <div className="text-gray-500 font-mono text-xs mb-4">/logs/dev-velocity</div>
              <div className="space-y-2 font-mono text-sm">
                <div className="text-primary font-bold">→ Feature generated... [0.2s]</div>
                <div className="text-primary font-bold">→ Tests generated... [0.1s]</div>
                <div className="text-red-400">→ Security Review... [PENDING]</div>
                <div className="text-red-500 font-bold">→ Error: 50+ files waiting for review</div>
              </div>
            </div>
          </div>

          {/* Right: Problem List */}
          <div className="space-y-px bg-gray-900 border-2 border-gray-900">
            {[
              {
                icon: <AlertTriangle className="w-6 h-6" />,
                title: "Speed over Security",
                desc: "AI coding tools prioritize working code, not secure code. They often suggest insecure patterns that work perfectly fine."
              },
              {
                icon: <Repeat className="w-6 h-6" />,
                title: "Zombie Bugs",
                desc: "You fix a bug, but re-generating the file brings it back. It's an endless cycle of finding the same issues."
              },
              {
                icon: <Clock className="w-6 h-6" />,
                title: "Manual Bottlenecks",
                desc: "Asking humans to review thousands of lines of machine-generated code is slow, boring, and error-prone."
              },
              {
                icon: <FileWarning className="w-6 h-6" />,
                title: "Tests get skipped",
                desc: "When you're moving fast, boring regression tests are the first thing to get cut. Security rots over time."
              }
            ].map((item, index) => (
              <div key={index} className="bg-white p-8 group hover:bg-gray-50 transition-colors">
                <div className="flex gap-6">
                  <div className="w-12 h-12 border-2 border-gray-900 flex items-center justify-center shrink-0 group-hover:border-red-600 group-hover:bg-red-600 transition-colors">
                    <div className="text-gray-900 group-hover:text-white transition-colors">
                      {item.icon}
                    </div>
                  </div>
                  <div>
                    <h3 className="font-bold text-xl mb-2 text-gray-900">{item.title}</h3>
                    <p className="text-gray-600 leading-relaxed">{item.desc}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default ProblemSection;