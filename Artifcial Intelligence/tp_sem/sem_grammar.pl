/*---------------------------------------------------------------*/
/* Telecom Paristech - J-L. Dessalles 2017                       */
/* NATURAL AND ARTIFICIAL INTELLIGENCE                           */
/*            http://teaching.dessalles.fr/NAI                   */
/*---------------------------------------------------------------*/



% ----------------------
% Phrase structure rules
% ----------------------

% Phrases have two arguments:
% xp(FXP, TXP)
% FXP is the feature structure, e.g. [num:sing, sentient:yes | _]
% TXP is the tree structure, used for display

% sentence

s(FVP,PS,TS) --> dp(FDP,PDP,TDP), vp(FVP,PVP,TVP), {	% determiner phrase + verb phrase
		att(FVP, subj, dp(FDP)),
		att(FDP, num, Num), 	% checking for number agreement 
		att(FVP, num, Num),		% between subject and verb
		link(_, PVP, PDP, PS),
		TS  = s(TDP,TVP) }.
s(FVP,PS,TS) --> dp(FDP,PDP,TDP), vp(FVP,PVP,TVP), {   
    att(FVP, subj, dp(FDP)),
    att(FDP, num, Num),     
    att(FVP, num, Num),        
    att(FVP, anim, A),
    att(FDP, anim, A),
    link(_, PVP, PDP, PS),
    TS = s(TDP,TVP) }.

s(FVP,PVP,TS) --> [it], vp(FVP,PVP,TVP), {	% impersonal verb  (it rains)
		att(FVP, subj, [it]),
		TS  = s(TVP) }.

% verb phrase

vp(FV,PV,TVP) --> v(FV,PV,TV), {			  % non transitive verb, eg. 'sleep'
		att(FV, cpl, []),
		TVP = vp(TV) }.

vp(FV,PVP,TVP) --> v(FV,PV,TV), dp(FDP,PDP,TDP), { % transitive verb, eg. 'like'
		att(FV, cpl, [dp(FDP)]),
		link(_, PV,PDP,PVP),
		TVP  = vp(TV,TDP) }.

vp(FV,PVP,TVP) --> v(FV,PV,TV), pp(FPP,PPP,TPP), { % transitive verb, indirect object: 'dream'
		att(FV, cpl, [pp(P)]),
		att(FPP, gloss, P),
		link(_, PV,PPP,PVP),
		TVP  = vp(TV,TPP) }.

vp(FV,PVP,TVP) --> v(FV,PV,TV), dp(FDP,PDP,TDP), dp(FDP2,PDP2,TDP2), { % ditransitive verb, eg. 'give'
		att(FV, cpl, [dp(FDP),dp(FDP2)]),
		link(_, PV,PDP,PV1),
		link(_, PV1,PDP2,PVP),
		TVP  = vp(TV, TDP, TDP2) }.

vp(FV,PVP,TVP) --> v(FV,PV,TV), cp(FCP,PCP,TCP), { % verb + cp: eg. 'dreams', 'believes'...
		att(FV, cpl, [cp(P)]),
		att(FCP, gloss, P),
		link(_, PV,PCP,PVP),
		TVP  = vp(TV, TCP) }.

vp(FV,PA,TVP) --> v(FV,_PV,TV), adj(_FADJ,PA,TADJ), { % verb + adj: eg. 'is', 'looks', 'seems'...
		att(FV, cpl, [adj(_)]),
		TVP  = vp(TV, TADJ) }.


% noun phrases
dp(FDP,PPN,TDP) --> pn(FDP,PPN,TPN), {	% proper noun
		TDP = dp(TPN) }.

dp(FN,PNP,TDP) --> det(FDET,TDET), np(FN,PNP,TN), { % 'the girl'
		att(FDET, num, Num), 
		att(FN, num, Num),
		TDP = dp(TDET,TN) }.

dp(FN,PNP,TDP) --> np(FN,PNP,TN), {	% 'girls'
		att(FN, num, plur), 
		TDP = dp(TN) }.


np(FN,PN,TNP) --> n(FN,PN,TN), {	% 'girl'
		TNP = np(TN) }.

np(FNP,PNP,TNP) --> adj(_FADJ,PA,TADJ), np(FNP,PNP1,TNP1), {	% 'nice girl'
		link(_,PA,PNP1,PNP),
		TNP = np(TADJ, TNP1) }.

np(FN,PNP,TNP) --> n(FN,PN,TN), pp(_FPP,PPP,TPP), {	% 'room of the girl'
		link(_,PN,PPP,PNP),
		TNP = np(TN, TPP) }.

% prepositional phrases
pp(FP, PDP,TPP) --> p(FP,TP), dp(_FDP,PDP,TDP), {
		TPP = pp(TP, TDP) }.

% subordinate clauses
cp(FC, PS,TCP) --> c(FC, TC), s(_FS, PS,TS), {
		TCP = cp(TC, TS) }.

% -------
% Lexicon
% -------
det([gloss:the, num:sing], det(the)) --> [the].
det([gloss:the, num:plur], det(the)) --> [the].
det([gloss:a,   num:sing], det(a))   --> [a].
det([gloss:all, num:plur], det(all)) --> [all].

n([gloss:child, num:sing], child(_), n(child)) --> [child].
%We set animate to the attribute anim to FV of [child]
n([gloss:child, num:sing, anim:animate], child(_), n(child)) --> [child].
n([gloss:child, num:plur], child(_), n(child)) --> [children].
n([gloss:game,  num:sing], game(_),  n(game))  --> [game].
n([gloss:game,  num:plur], game(_),  n(game))  --> [games].
n([gloss:girl,  num:sing], girl(_),  n(girl))  --> [girl].
n([gloss:girl,  num:plur], girl(_),  n(girl))  --> [girls].
n([gloss:boy,   num:sing], boy(_),   n(boy))   --> [boy].
n([gloss:boy,   num:plur], boy(_),   n(boy))   --> [boys].
n([gloss:room,  num:sing], room(_),  n(room))  --> [room].
%The same for FV of [room] but setting inanimate for anim
n([gloss:room, num:sing, anim:inanimate], room(_), n(room)) --> [room].
n([gloss:house, num:plur], house(_), n(house)) --> [house].
n([gloss:hall, num:plur],  hall(_),  n(hall))  --> [hall].
n([gloss:garden, num:plur],garden(_),n(garden)) --> [garden].

% proper nouns
pn([gloss:john, num:sing], john, pn(john)) --> ['John'].
pn([gloss:pat,  num:sing], pat, pn(pat))   --> ['Pat'].
pn([gloss:mary, num:sing], mary, pn(mary)) --> ['Mary'].
pn([gloss:ann,  num:sing], ann, pn(ann))   --> ['Ann'].

% verbs
v([gloss:sleep,  num:sing,pers:3,subj:dp(_), cpl:[]], sleep(_), v(sleep)) --> [sleeps].
%Agreement is checked for sleeps
v([gloss:sleep,  num:sing,pers:3,subj:dp(_|anim:animate|_), cpl:[]], sleep(_), v(sleep)) --> [sleeps].
v([gloss:sleep,  num:plur,pers:3,subj:dp(_), cpl:[]], sleep(_), v(sleep)) --> [sleep].

v([gloss:play,  num:sing,pers:3,subj:dp(_),  cpl:[]], play(_), v(play)) --> [plays].
v([gloss:play,  num:plur,pers:3,subj:dp(_),  cpl:[]], play(_), v(play)) --> [play].

v([gloss:like,  num:sing,pers:3,subj:dp(_),  cpl:[dp(_)]], like(_,_), v(like)) --> [likes].
v([gloss:like,  num:plur,pers:3,subj:dp(_),  cpl:[dp(_)]], like(_,_), v(like)) --> [like].

v([gloss:dream,num:sing,pers:3,subj:dp(_),   cpl:[pp(of)]],   dream(_,_),   v(dream)) --> [dreams].
v([gloss:dream,num:plur,pers:3,subj:dp(_),   cpl:[pp(of)]],   dream(_,_),   v(dream)) --> [dream].
v([gloss:dream,num:sing,pers:3,subj:dp(_),   cpl:[cp(that)]], dream(_,_),   v(dream)) --> [dreams].
v([gloss:dream,num:plur,pers:3,subj:dp(_),   cpl:[cp(that)]], dream(_,_),   v(dream)) --> [dream].

v([gloss:believe,num:sing,pers:3,subj:dp(_), cpl:[cp(that)]], believe(_,_), v(believe)) --> [believes].
v([gloss:believe,num:plur,pers:3,subj:dp(_), cpl:[cp(that)]], believe(_,_), v(believe)) --> [believe].

v([gloss:give, num:sing,pers:3,subj:dp(_),   cpl:[dp(_),dp(_)]],give(_,_,_), v(give)) --> [gives].
v([gloss:give, num:plur,pers:3,subj:dp(_),   cpl:[dp(_),dp(_)]],give(_,_,_), v(give)) --> [give].

v([gloss:talk,num:sing,pers:3,subj:dp(_),    cpl:[pp(with), pp(about)]], talk(_,_,_), v(talk)) --> [talks].
v([gloss:talk,num:plur,pers:3,subj:dp(_),    cpl:[pp(with), pp(about)]], talk(_,_,_), v(talk)) --> [talk].


v([gloss:look,num:sing,pers:3,subj:dp(_), cpl:[adj(_)]], look, v(look)) --> [looks].
v([gloss:look,num:plur,pers:3,subj:dp(_), cpl:[adj(_)]], look, v(look)) --> [look].

v([gloss:be,num:sing,pers:3,subj:dp(_),   cpl:[adj(_)]], be, v(be)) --> [is].
v([gloss:be,num:plur,pers:3,subj:dp(_),   cpl:[adj(_)]], be, v(be)) --> [are].
v([gloss:be,num:sing,pers:3,subj:dp(_),   cpl:[pp(in)]], be, v(be)) --> [is].
v([gloss:be,num:plur,pers:3,subj:dp(_),   cpl:[pp(in)]], be, v(be)) --> [are].

v([gloss:rain,num:sing,pers:3, subj:[it],  cpl:[]], [rain], v(rain)) --> [rains].

% adj.
adj([gloss:nice],  nice(_),  adj(nice))  --> [nice].
adj([gloss:small], small(_), adj(small)) --> [small].
adj([gloss:large], large(_), adj(large)) --> [large].
adj([gloss:big],   big(_),   adj(big))	 --> [big].
adj([gloss:happy], happy(_), adj(happy)) --> [happy].
adj([gloss:quiet], quiet(_), adj(quiet)) --> [quiet].

% prep
p([gloss:on],   p(on))	   --> [on].
p([gloss:in],   p(in))	   --> [in].
p([gloss:with], p(with))   --> [with].
p([gloss:about],p(about))  --> [about].
p([gloss:of],   p(of))	   --> [of].
p([gloss:to],   p(to))	   --> [to].

% complementizer
c([gloss:that], c(that))   --> [that].
c([gloss:when], c(when))   --> [when].
