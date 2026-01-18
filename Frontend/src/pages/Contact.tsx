import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { Mail, MessageSquare, MapPin } from "lucide-react";

const Contact = () => {
    return (
        <div className="min-h-screen bg-background flex flex-col">
            <Navbar />
            <main className="flex-grow pt-24 pb-20">
                <div className="container mx-auto px-6">
                    <div className="grid lg:grid-cols-2 gap-20">
                        {/* Info */}
                        <div className="space-y-10">
                            <div>
                                <h1 className="font-display text-4xl lg:text-5xl font-bold mb-6">Talk to our security engineers.</h1>
                                <p className="text-xl text-muted-foreground leading-relaxed">
                                    We help teams of all sizes secure their infrastructure.
                                    Whether you need a custom enterprise plan or technical support, we're here.
                                </p>
                            </div>

                            <div className="space-y-6">
                                {[
                                    { icon: <Mail className="w-5 h-5" />, label: "Email Us", value: "security@sentinel.dev" },
                                    { icon: <MessageSquare className="w-5 h-5" />, label: "Live Chat", value: "Available 9am-5pm EST" },
                                    { icon: <MapPin className="w-5 h-5" />, label: "HQ", value: "San Francisco, CA" },
                                ].map((item) => (
                                    <div key={item.label} className="flex items-center gap-4 p-4 border border-border rounded-xl bg-white">
                                        <div className="w-10 h-10 rounded-full bg-secondary flex items-center justify-center text-foreground">
                                            {item.icon}
                                        </div>
                                        <div>
                                            <div className="text-xs text-muted-foreground uppercase tracking-wider font-bold mb-0.5">{item.label}</div>
                                            <div className="font-medium text-foreground">{item.value}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Form */}
                        <div className="bg-white p-8 rounded-2xl border border-border shadow-sm">
                            <form className="space-y-6">
                                <div className="grid md:grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-foreground">First Name</label>
                                        <input type="text" className="w-full px-4 py-3 rounded-lg border border-border focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all bg-background" placeholder="Jane" />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-sm font-bold text-foreground">Last Name</label>
                                        <input type="text" className="w-full px-4 py-3 rounded-lg border border-border focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all bg-background" placeholder="Doe" />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-bold text-foreground">Work Email</label>
                                    <input type="email" className="w-full px-4 py-3 rounded-lg border border-border focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all bg-background" placeholder="jane@company.com" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-sm font-bold text-foreground">Message</label>
                                    <textarea className="w-full px-4 py-3 rounded-lg border border-border focus:border-primary focus:ring-1 focus:ring-primary outline-none transition-all bg-background min-h-[150px]" placeholder="Tell us about your security needs..."></textarea>
                                </div>
                                <button type="button" className="w-full btn-primary py-4 text-base">
                                    Send Message
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default Contact;
