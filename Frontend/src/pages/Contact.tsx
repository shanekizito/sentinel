import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import { Mail, MessageSquare, MapPin, ArrowRight } from "lucide-react";

const Contact = () => {
    return (
        <div className="min-h-screen bg-white flex flex-col font-sans">
            <Navbar />
            <main className="flex-grow pt-32 pb-20">
                <div className="container mx-auto px-6">
                    <div className="grid lg:grid-cols-2 gap-px bg-gray-200 border-2 border-gray-200">
                        {/* Info Side */}
                        <div className="bg-white p-12 lg:p-20 flex flex-col justify-between">
                            <div className="space-y-8">
                                <h1 className="font-display text-5xl font-bold bg-white text-gray-900 leading-tight">
                                    Talk to our <br />
                                    <span className="text-primary">security engineers.</span>
                                </h1>
                                <p className="text-xl text-gray-600 leading-relaxed font-light border-l-2 border-gray-100 pl-6">
                                    We help teams of all sizes secure their infrastructure.
                                    Whether you need a custom enterprise plan or technical support, we're here.
                                </p>
                            </div>

                            <div className="space-y-6 mt-16">
                                {[
                                    { icon: <Mail className="w-5 h-5" />, label: "Email Us", value: "security@sentinel.dev" },
                                    { icon: <MessageSquare className="w-5 h-5" />, label: "Live Chat", value: "Available 9am-5pm EST" },
                                    { icon: <MapPin className="w-5 h-5" />, label: "HQ", value: "San Francisco, CA" },
                                ].map((item) => (
                                    <div key={item.label} className="flex items-center gap-6 group">
                                        <div className="w-12 h-12 border-2 border-gray-900 flex items-center justify-center text-gray-900 group-hover:bg-primary group-hover:border-primary group-hover:text-white transition-colors">
                                            {item.icon}
                                        </div>
                                        <div>
                                            <div className="text-xs text-gray-400 uppercase tracking-widest font-bold mb-1">{item.label}</div>
                                            <div className="font-bold text-gray-900 text-lg">{item.value}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Form Side */}
                        <div className="bg-gray-50 p-12 lg:p-20 relative overflow-hidden">
                            {/* Grid Background */}
                            <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,0,0,0.05)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,0,0,0.05)_1px,transparent_1px)] bg-[size:2rem_2rem] pointer-events-none" />

                            <form className="space-y-8 relative z-10 max-w-md mx-auto">
                                <div className="grid md:grid-cols-2 gap-8">
                                    <div className="space-y-2">
                                        <label className="text-xs font-bold uppercase tracking-widest text-gray-500">First Name</label>
                                        <input type="text" className="w-full px-4 py-4 bg-white border-2 border-gray-200 focus:border-gray-900 outline-none transition-colors placeholder:text-gray-300 rounded-none text-gray-900 font-bold" placeholder="JANE" />
                                    </div>
                                    <div className="space-y-2">
                                        <label className="text-xs font-bold uppercase tracking-widest text-gray-500">Last Name</label>
                                        <input type="text" className="w-full px-4 py-4 bg-white border-2 border-gray-200 focus:border-gray-900 outline-none transition-colors placeholder:text-gray-300 rounded-none text-gray-900 font-bold" placeholder="DOE" />
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold uppercase tracking-widest text-gray-500">Work Email</label>
                                    <input type="email" className="w-full px-4 py-4 bg-white border-2 border-gray-200 focus:border-gray-900 outline-none transition-colors placeholder:text-gray-300 rounded-none text-gray-900 font-bold" placeholder="jane@company.com" />
                                </div>
                                <div className="space-y-2">
                                    <label className="text-xs font-bold uppercase tracking-widest text-gray-500">Message</label>
                                    <textarea className="w-full px-4 py-4 bg-white border-2 border-gray-200 focus:border-gray-900 outline-none transition-colors placeholder:text-gray-300 rounded-none min-h-[150px] text-gray-900 font-bold resize-none" placeholder="Tell us about your security needs..."></textarea>
                                </div>
                                <button type="button" className="w-full h-16 bg-primary text-white font-bold uppercase tracking-widest hover:bg-primary/90 transition-all flex items-center justify-center gap-3 text-sm shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] hover:shadow-none hover:translate-x-[2px] hover:translate-y-[2px]">
                                    Send Message <ArrowRight className="w-5 h-5" />
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
