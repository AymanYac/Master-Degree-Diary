/*---------------------------------------------------------------*/
/* Telecom Paristech - J-L. Dessalles 2017                       */
/* NATURAL AND ARTIFICIAL INTELLIGENCE                           */
/*            http://teaching.dessalles.fr/NAI                   */
/*---------------------------------------------------------------*/



% If necessary, declare a new operator, ':', which will be used as separator between attribute and value
% :- op(600,xfy,':').

% -- att/3 returns the value of a substructure
% att([a:b,c:d],c,X), returns X=d.
att([F:U|_], F, U1):- 
	!,
	U = U1.	% if the attribute is present, values must match

att([_|R], F, U) :-
	att(R, F, U).

att([], _F, _U).	% success if the attribute is absent


% ------------------------------------------------------
% Don't change anything below this line... unless you 
% know what you are doing
% ------------------------------------------------------

% get_lines makes the conversion between strings and list
% 
get_line(Phrase) :-
        collect_wd(String), 
        str2wlist(Phrase, String).

collect_wd([C|R]) :-
        get0(C), not(member(C, [-1, 10, 13, 27])), !, 
        collect_wd(R).
collect_wd([]).

str2wlist(Phrase,Str) :-
	atom_codes(A,Str),
	atomic_list_concat(Phrase,' ',A).

% print_tree prints out a syntactic tree based on the
% parentheric expression.
% Indent
print_tree(StructPhrase) :-
	nl,
	print_tree('         ','         ',StructPhrase),
	nl,nl.

print_tree(_,_,StructPhrase) :-
	StructPhrase =.. [LibPhrase],
	/* il s'agit d'un terminal */
	!,
	print_phr([' : ',LibPhrase]).
print_tree(Indent,Prefixe,StructPhrase) :-
	%StructPhrase =.. [LibPhrase,AttrPhrase|SousStruct],
	%nl,print_phr([Prefixe,LibPhrase,AttrPhrase]),
	StructPhrase =.. [LibPhrase|SousStruct],
	nl,print_phr([Prefixe,LibPhrase]),
	print_node(Indent,SousStruct).

print_node(_,[]).
print_node(Indent,[SP]) :-
	% c'est le dernier fils, on ne dessine plus
	% la branche parente                        
	!,
	atom_concat(Indent,'   ',NewIndent),
	atom_concat(Indent,'  |__',IndentLoc),
	print_tree(NewIndent,IndentLoc,SP).
print_node(Indent,[SP|SPL]) :-
	atom_concat(Indent,'  |',NewIndent),
	atom_concat(Indent,'  |__',IndentLoc),
	print_tree(NewIndent,IndentLoc,SP),
	print_node(Indent,SPL).

print_phr([]).
print_phr([S|Sl]) :-
	write(S),
	print_phr(Sl).


