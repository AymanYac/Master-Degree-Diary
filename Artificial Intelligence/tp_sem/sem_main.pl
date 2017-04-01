/*---------------------------------------------------------------*/
/* Telecom Paristech - J-L. Dessalles 2017                       */
/* NATURAL AND ARTIFICIAL INTELLIGENCE                           */
/*            http://teaching.dessalles.fr/NAI                   */
/*---------------------------------------------------------------*/



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main file of procedural semantics           %
% - loads other modules                       %
% - runs parsing                              %
% - operate semantic linking                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



:- consult('sem_util.pl').		% loading utility predicates
								% these predicates include 'get_line', 'att' and 'print_tree'

:- consult('sem_grammar.pl').	% loading grammar

:- consult('sem_world.pl').		% loading world knowledge


% Main goal - starts the DCG parser
go :-
	nl, write('Sentence > '),
	get_line(Sentence),
	Sentence \== [''],
	!,
	dcg_parse(Sentence).	% start DCG parser
go.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% parsing                                     %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dcg_parse(L) :-

	%%%%%%%%%%%%%%% running DCG parser  %%%%%%%%%%%%%%
	s(FS,Pred,Tree,L,[]),
	nl, write('The sentence is syntactically correct'),nl,

	%%%%%%%%%%%%%%% Printing results	%%%%%%%%%%%%%%
	write(FS), nl,
	print_tree(Tree),		% comment if tree display is annoying
	Pred = [Predicate|Constraints],
	write('--> '), write(Predicate), write('\t\t'), write(Constraints), nl,
	test(Predicate),
	write('this sentence makes sense'),nl,
	get_single_char(C), member(C, [27]), !, nl,  % type 'Esc' to skip alternatives
	go.
dcg_parse(_) :-
	go.

	
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% semantic linking                            %
% ----------------                            %
% two connected phrases must share a variable %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% link(_,A,_B,[A]) :- !.	% comment this line to unveil the next ones

% dealing with lists firt
link(NroArg, [Pred1| Preds], Pred2, Preds1) :-
	!,
	link(NroArg, Pred1, Pred2, Pred),
	append(Pred, Preds, Preds1).
link(NroArg, Pred1, [Pred2| Preds], Preds1) :-
	!,
	link(NroArg, Pred1, Pred2, Pred),
	append(Pred, Preds, Preds1).

% proper semantic linking
link(NroArg, Pred1, Pred2, [Pred1,Pred2]) :-
	Pred1 =.. [_ | Args1],		% converts predicate into list to extracts arguments.  p(a,b,c) =.. [p,a,b,c]
	nth1(NroArg, Args1, A),		% selects the nth argument
	Pred2 =.. P_Arg2,			% converts predicate into list
	select_arg(P_Arg2, A),	% selects an argument
	execute(Pred2).


select_arg([P], P).			% no arguments, this is a constant (e.g. proper noun)
select_arg([_, A | _], A).	% only first argument considered

	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 'execute(P)' tries to run predicate P
% if no program corresponds to P, keeps P as such
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

execute(A) :- atom(A), !.	% constants are not executed (e.g. proper noun)
execute(P) :-
	% write('executing '), write(P),nl,
	P =.. [Pred | Args],	% separating predicate from arguments
	length(Args, Arity),	% Arity = number of arguments
	dynamic(Pred/Arity),	% declare the predicate as dynamic (useful in SWI-Prolog)
	P.	%%%%%%%%%%  Exectution
execute(P) :-
	% P cannot be executed. It is kept as such, but only if there is no program that corresponds to P
	P =.. [Pred | Args],	% separating predicate from arguments
	length(Args, Arity),	% Arity = number of arguments in P
	length(VoidArgs, Arity),% Creates a list of variables of same length
	P0 =.. [Pred | VoidArgs],	% replaces constants by variables in P
	not(P0),	% no existing program for P0
	!.

test(P) :-	
	execute(P).
test(_P) :-
	write('    can''t find any (further) meaning'), nl, 
	fail.
