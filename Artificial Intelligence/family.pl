
% partial elementary English grammar

% --- Grammar
s --> np, vp.
np --> det, n.		% Simple noun phrase
np --> det, n, pp.		% Noun phrase + prepositional phrase 
np --> [kirk].
vp --> v.           % Verb phrase, intransitive verb
vp --> v, np.		% Verb phrase, verb + complement:  like X
vp --> v, pp.		% Verb phrase, verb + indirect complement : think of X 
vp --> v, np, pp.	% Verb phrase, verb + complement + indirect complement : give X to Y 
vp --> v, pp, pp.	% Verb phrase, verb + indirect complement + indirect complement : talk to X about Y
pp --> p, np.		% prepositional phrase

% -- Lexicon
det --> [the].
det --> [my].
det --> [her].
det --> [his].
det --> [a].
det --> [some].
n --> [dog].
n --> [daughter].
n --> [son].
n --> [sister].
n --> [aunt].
n --> [neighbour].
n --> [cousin].
v --> [grumbles].
v --> [likes].
v --> [gives].
v --> [talks].
v --> [annoys].
v --> [hates].
v --> [cries].
v --> [sleeps].
p --> [of].
p --> [to].
p --> [about].


s([number:N,person:P,gender:G,sentience:S]) --> np([number:N,person:P,gender:G,sentience:S]),vp([number:N,person:P,gender:G,sentience:S]).

np([number:sing, person:3, gender:feminine, sentience:true]) --> [mary].
np([number:sing, person:3, gender:masculine, sentience:true]) --> [john].
np([number:plur, person:3, gender:_, sentience:true]) --> [people].
np([number:sing, person:3, gender:_, sentience:false]) -->[rock].
np([number:plur, person:3, gender:_, sentience:false]) -->[rocks].

vp([number:N,person:P,gender:G,sentience:S]) --> v([subj:[number:N,person:P,gender:G,sentience:S], event:E]).

v([subj:[number:sing, person:3, gender:_, sentience:true], event:false]) --> [thinks],.
v([subj:[number:plur, person:3, gender:_, sentience:true], event:false]) --> [think].
v([subj:[number:sing, person:3, gender:_, sentience:_], event:true]) --> [falls].
v([subj:[number:plur, person:3, gender:_, sentience:_], event:true]) --> [fall].
v([subj:[number:sing, person:3, gender:_, sentience:false], event:true]) --> [erodes].
v([subj:[number:plur, person:3, gender:_, sentience:false], event:true]) --> [erode].

