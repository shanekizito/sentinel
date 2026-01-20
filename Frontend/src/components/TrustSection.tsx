import { Shield, Eye, Lock, FileCode, UserCheck } from "lucide-react";

const guarantees = [
  {
    icon: <Shield className="w-5 h-5" />,
    title: "Your IP is Yours",
    description: "We never train models on your proprietary code."
  },
  {
    icon: <Lock className="w-5 h-5" />,
    title: "Runs Locally",
    description: "Option to run entirely within your own private network."
  },
  {
    icon: <Eye className="w-5 h-5" />,
    title: "Read-Only Access",
    description: "We can't commit changes unless you explicitly approve them."
  },
  {
    icon: <FileCode className="w-5 h-5" />,
    title: "Transparent Diffs",
    description: "See exactly what changed, down to the character."
  },
  {
    icon: <UserCheck className="w-5 h-5" />,
    title: "You're in Control",
    description: "Human approval is always the final step."
  }
];

const TrustSection = () => {
  return (
    <section className="py-24 bg-white border-y-2 border-gray-900 relative overflow-hidden">
      {/* Grid Lines */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute left-[25%] top-0 bottom-0 w-px bg-gray-100" />
        <div className="absolute right-[25%] top-0 bottom-0 w-px bg-gray-100" />
      </div>

      <div className="container mx-auto px-6 relative z-10">
        <div className="max-w-4xl mb-16 border-l-4 border-gray-900 pl-8">
          <h2 className="font-display text-4xl font-bold text-gray-900 mb-4">
            Built on Trust.
          </h2>
          <p className="text-xl text-gray-600">
            We know your code is your most valuable asset. We treat it that way.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-px bg-gray-900 border-2 border-gray-900 max-w-5xl">
          {guarantees.map((item, index) => (
            <div
              key={item.title}
              className="bg-white p-8 group hover:bg-gray-50 transition-colors"
            >
              <div className="w-10 h-10 border-2 border-gray-900 flex items-center justify-center mb-4 group-hover:border-primary group-hover:bg-primary transition-colors">
                <div className="text-gray-900 group-hover:text-white transition-colors">
                  {item.icon}
                </div>
              </div>
              <h3 className="font-display font-bold text-lg mb-2 text-gray-900">{item.title}</h3>
              <p className="text-sm text-gray-600 leading-relaxed">{item.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default TrustSection;