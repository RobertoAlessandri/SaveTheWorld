s.boot;

//n = NetAddr.new("127.0.0.1", 7771);

(
n = NetAddr.new("127.0.0.1", 7771);
//NetAddr.langPort;    // retrieve the current port SC is listening to
thisProcess.openUDPPort(7771); // attempt to open 7771
o = OSCFunc({ arg msg, time, addr, recvPort = 7771; [msg[0]].postln}, '/soundscape_modifier', n);
(
f = { |msg, time, addr, recvPort = 7771|
	addr = n;
	if(msg[0] != '/status.reply') {
		if(msg[0] != '/globe_control') {
			if(msg[0] == '/soundscape_modifier') {
				//"received message % named % with value % from % on port%\n".postf(msg, msg[0], msg[1], recvPort);
				//~sound.set(\amp, msg[1]);
				//~x[0].set(\amp, 0.2 * (1 - msg[1]));
				//~x[1].set(\amp, 0.2 * (1 - msg[1]));
				//~x[2].set(\amp, 0.2 * (1 - msg[1]));
				//~x[3].set(\amp, 0.2 * (1 - msg[1]));



			}
		}
	}
};
thisProcess.addOSCRecvFunc(f);
);
)

(
b = Buffer.alloc(s,512,1,{|z|z.sine1Msg(1.0/[1,3,5,7,9,11,13,15,17])});
c = Buffer.read(s, "/path"); // Not needed a path, but still the variable c is necessary

fork{
	s.sync;
	~sound = {arg amp = 1;
		var sig;
		var local;
		// TCHoose select randomly an output based on a (trigger, array of inputs)
		var f = [30,60,15]*Lag.kr(TChoose.kr(Impulse.kr(0.05),[0.75, 0.5, 1]), 8);
		// COsc is a chorus COsc.ar(bufnum, frequency, beat frequency, multiplier, adder)
		sig = Mix(COsc.ar(b.bufnum, f + SinOsc.ar(f*25, 0, LFTri.kr(0.01).range(0, 10)), [0.1, 0.1001, 0.2], 0.2))*0.1;
		sig = sig; //?
		// LeakDC is a linear filter that removes DC (in, coeff, mul, add)
		// Ringz is a 2-pole resonant filter (input, resonant frequencym, decay to -60 dB, mul, add)


		// LFTri is A non-band-limited triangle oscillator with  output ranges from -1 to +1 (freq, phase, mul, add)
		sig = LeakDC.ar(Ringz.ar(sig, TChoose.kr(Impulse.kr(0.1),[88, 97, 99, 100].midicps), LFTri.kr([0.05, 0.051]).range(0.2, 0.5)));
		// LPF is a II order LPF (input, cutoff frequency, mul, add)
		sig = LPF.ar(sig, 3000*LFTri.kr(0.01).range(0.1, 1));
		// Splay spreads an array of channels across the stereo field
		sig = sig + (Splay.ar(Array.fill(4, {PlayBuf.ar(2, c, rrand(-0.8, 0.8), loop: 2)*0.01}), 0.5));
		sig = CombC.ar(sig, 1.0, [0.1, 0.2], LFTri.kr(0.05).range(5, 9));

		local = sig + LocalIn.ar(2);
		15.do({
			local = AllpassN.ar(local, 0.06, Rand(0.001, 0.06), 3)
		});
		LocalOut.ar(local*0.4);

		Out.ar(0, Limiter.ar(LPF.ar(local * amp, 4000), 0.8)*EnvGen.kr(Env([0, 1, 1, 0],[9, 240, 30])));
	}.play;

}
)

(
SynthDef(\saw,{
	arg freq=432, amp=0.2, gate=1, out=0, feedback=1;
	var env, sig, gVerb, gVerb2, sig2, del, local, gDelay;
	env = EnvGen.kr(
		Env([0, 1, 1, 0],[9, 240, 30]),
		gate,
		doneAction: 2);
	sig = VarSaw.ar(freq)*env*amp;
	gVerb = FreeVerb.ar(sig, mix: 1, room:1,damp: 1, mul: 1.0);
	gVerb2 = FreeVerb.ar(gVerb, mix: 1, room:1,damp: 1, mul: 1.0);
	del = CombN.ar(gVerb2, maxdelaytime: 0.5, delaytime: 0.3, decaytime: 1.0, mul: 1.0, add: 0.0);
	gDelay = DelayC.ar(del, maxdelaytime: 0.9, delaytime: 0.7, mul: 1.0, add: 0.0);
	15.do{local=AllpassN.ar(gVerb,0.06,Rand(0.001,0.06),3)};
	LocalOut.ar(sig2*feedback);
	sig = Splay.ar(gDelay);
	gVerb = FreeVerb.ar(sig, mix: 0.5, room:0.8,damp: 0.9, mul: 1.0, add: 0.0);
	Out.ar([0,1],sig);
}).play;
)

(
~x=[45,61,52,68].collect({
	arg note, index, ampp = 0.1;
	Synth(\saw,[\freq, note.midicps, \amp, ampp]);
	//FreeVerb.new(temp, mix: 0.33, room: 0.5, damp: 0.5, mul: 1.0, add: 0.0)
	//FreeVerb.new(temp, mix: 0.33, room: 0.5, damp: 0.5, mul: 1.0, add: 0.0);
})
)

s.quit;
