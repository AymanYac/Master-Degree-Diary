/*---------------------------------------------------------------*/
/* PARISTECH - Ecole Nationale Superieure des Telecommunications */
/*---------------------------------------------------------------*/
/* J-L. Dessalles 2009                               Dep. INFRES */
/*                                                               */
/* >>>>>  Logic programming and Knowledge representation  <<<<<< */
/*              teaching.dessalles.fr/LKR                        */
/*                                                               */
/*---------------------------------------------------------------*/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Conversion from DCG to 'rule' %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  s --> gn, gv.
%  is converted into:
%  rule(s,[gn,gv]).
%

dcg2rules(DCGFile) :-
    retractall(rule(_,_)),
    consult(DCGFile),
    member(Head, [s, np, vp, pp, n, det, v, p]),  % list of non-terminals
    HeadPredicate =.. [Head,Input,_],
    clause(HeadPredicate,RightHandSide),    % retrieves DCG in clause form
    disjunction(RightHandSide, RHSL),    % process disjunctive DCG (those that use ';')
	member(RHS, RHSL),
    queue2list(Input, RHS, RHSinListForm),       % converts RHS sequence into list
    assert(rule(Head, RHSinListForm)),
    fail.
dcg2rules(_).

disjunction((RHHD;RHSQ), [RHHD|RHS]) :-   % disjunctive clause queues are like '(Queue1; OhterQueues)'
    !, 
	disjunction(RHSQ, RHS).
disjunction(RH, [RH]).

queue2list([Lexeme|_], true, [Lexeme]).		% special case of empty queues
queue2list(_, (HP,Q),[H|QL]) :-   % conjunctive clause queues are like '(Queue1, OhterQueues)'
    !,
    HP =.. [H,_,_],     % get rid of phantom arguments
    queue2list(_, Q,QL).
queue2list(_, HP, [H]) :-
    HP =.. ['=',_,[H|_]],   % because lexical rules n --> [L] are stored as 'X = [L|_]'
    !.
queue2list(_, HP, [H]) :-
    HP =.. [H,_,_],
	!.
	%%%%%%%%%%%%%
	% bottom-up recognition  %
	%%%%%%%%%%%%%

:- consult('dcg2rules.pl').     % DCG to 'rule' converter: np --> det, n. becomes rule(gn,[det,n])
:- dcg2rules('family.pl').      % performs the conversion by asserting rule(np,[det,n])


go :-
	bup([the,sister,talks,about,her,cousin]).

bup([s]).  % success when one gets s after a sequence of transformations
bup(P):-
	write(P), nl, % get0(_),
	append(Pref, Rest, P),   % P is split into three pieces 
	append(RHS, Suff, Rest), % P = Pref + RHS + Suff
	rule(X, RHS),	% bottom up use of rule
	append(Pref, [X|Suff], NEWP),  % RHS is replaced by X in P:  NEWP = Pref + X + Suff
	bup(NEWP).  % lateral recursive call

	