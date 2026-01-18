import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Check, ArrowUpRight } from "lucide-react";
import { useState } from "react";

const PricingSection = () => {
  const [billingPeriod, setBillingPeriod] = useState<"monthly" | "annually">("monthly");

  const basicFeatures = [
    "Real-time security updates for major cyber threats",
    "Comprehensive cyber security solutions for safeguarding digital assets",
    "Security notifications for up to 5 times",
    "Access to the community forum",
    "Basic analytics and charting tools",
    "Email support",
  ];

  const professionalFeatures = [
    "All features of the Premium Plan",
    "Professional-grade analytics and charting tools",
    "API access for custom integrations",
    "Unlimited price alerts and advanced alerts",
    "Personalized market reports and investment recommendations",
    "Dedicated account manager",
  ];

  return (
    <section className="relative min-h-screen py-24 overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-background" />
      <div className="absolute inset-0 bg-grid-cyber" />
      
      {/* Gradient glow effects */}
      <div className="absolute top-1/4 left-1/4 w-[600px] h-[600px] rounded-full bg-primary/5 blur-[150px]" />
      <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] rounded-full bg-primary/5 blur-[120px]" />
      
      {/* Floating decorative squares */}
      <div className="floating-square w-8 h-8 top-32 left-[10%] animate-float-slow opacity-60" />
      <div className="floating-square w-6 h-6 top-48 right-[15%] animate-float-medium opacity-40" style={{ animationDelay: "1s" }} />
      <div className="floating-square w-10 h-10 bottom-32 left-[20%] animate-float-slow opacity-50" style={{ animationDelay: "2s" }} />
      <div className="floating-square w-5 h-5 top-[40%] right-[8%] animate-float-medium opacity-50" style={{ animationDelay: "0.5s" }} />
      <div className="floating-square w-7 h-7 bottom-[20%] right-[25%] animate-float-slow opacity-40" style={{ animationDelay: "1.5s" }} />
      
      <div className="container mx-auto px-6 relative z-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <div className="inline-flex items-center gap-2 text-primary text-sm font-medium mb-6">
            <span>Pricing Plans</span>
            <ArrowUpRight className="w-4 h-4" />
          </div>
          <h2 className="font-display text-display-md md:text-display-lg text-foreground mb-4">
            Unlock Your Security Potential with
            <br />
            Tokematic Services
          </h2>
        </motion.div>

        <div className="grid lg:grid-cols-12 gap-8 items-start">
          {/* Left sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="lg:col-span-3"
          >
            <h3 className="font-display text-2xl md:text-3xl text-foreground mb-4 leading-tight">
              Choose the Perfect Plan for Your Security Needs
            </h3>
            <p className="text-muted-foreground text-sm mb-8 leading-relaxed">
              Discover our adaptable security solutions and choose the one that perfectly fits your cyber security requirements.
            </p>
            
            {/* Billing Toggle */}
            <div className="inline-flex items-center bg-secondary/50 rounded-full p-1">
              <button
                onClick={() => setBillingPeriod("monthly")}
                className={`px-5 py-2 rounded-full text-sm font-medium transition-all ${
                  billingPeriod === "monthly"
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Monthly
              </button>
              <button
                onClick={() => setBillingPeriod("annually")}
                className={`px-5 py-2 rounded-full text-sm font-medium transition-all ${
                  billingPeriod === "annually"
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Annually
              </button>
            </div>
          </motion.div>

          {/* Pricing Cards */}
          <div className="lg:col-span-9 grid md:grid-cols-2 gap-6">
            {/* Basic Plan */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="cyber-card pricing-card-gradient p-6 rounded-2xl hover-lift"
            >
              <div className="mb-6">
                <h4 className="font-display text-xl text-foreground mb-4">Basic Plan</h4>
                <div className="flex items-baseline gap-1 mb-3">
                  <span className="font-display text-4xl text-foreground">$15</span>
                  <span className="text-muted-foreground text-sm">/year</span>
                </div>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  Individuals interested in enhancing their cyber security skills.
                </p>
              </div>

              <div className="mb-6">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-4">
                  What you will get
                </p>
                <ul className="space-y-3">
                  {basicFeatures.map((feature, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <Check className="w-3 h-3 text-primary" />
                      </div>
                      <span className="text-sm text-muted-foreground leading-relaxed">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <Button 
                className="w-full bg-primary text-primary-foreground hover:bg-primary/90 font-medium py-5 glow-green-sm hover:glow-green transition-all"
              >
                Book a Demo
              </Button>
            </motion.div>

            {/* Professional Plan */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 0.3 }}
              className="cyber-card pricing-card-gradient p-6 rounded-2xl hover-lift"
            >
              <div className="mb-6">
                <h4 className="font-display text-xl text-foreground mb-4">Professional Plan</h4>
                <div className="flex items-baseline gap-1 mb-3">
                  <span className="font-display text-4xl text-foreground">$60.00</span>
                  <span className="text-muted-foreground text-sm">/month</span>
                </div>
                <p className="text-muted-foreground text-sm leading-relaxed">
                  Cybersecurity experts and institutions seeking robust protection tools.
                </p>
              </div>

              <div className="mb-6">
                <p className="text-xs text-muted-foreground uppercase tracking-wider mb-4">
                  What you will get
                </p>
                <ul className="space-y-3">
                  {professionalFeatures.map((feature, index) => (
                    <li key={index} className="flex items-start gap-3">
                      <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                        <Check className="w-3 h-3 text-primary" />
                      </div>
                      <span className="text-sm text-muted-foreground leading-relaxed">{feature}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <Button 
                className="w-full bg-primary text-primary-foreground hover:bg-primary/90 font-medium py-5 glow-green-sm hover:glow-green transition-all"
              >
                Book a Demo
              </Button>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default PricingSection;
