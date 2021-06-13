:- dynamic position/2.
:- dynamic positioned/2.
:- dynamic quest/2.
:- dynamic day/2.
:- dynamic closed/2.
:- dynamic bag/1.

%---------------Game loop and start----------------
start :- quest(no,N), N =:= 0, N1 is N + 1, 
	retract(quest(no,N)), 
	assert(quest(no,N1)), 
	write(' You woke up in the moring. Your head hurts and humms. You do not know where you are or how did you get here. You need to find a way out of this place.'),
	nl,
	look,
	repeat,
	nl,
    write('> '),
    read(X),
    call(X),
    fail.
start :- write(' Game already started ').
newGame :-  quest(no,N), retract(quest(no,N)), assert(quest(no,0)), start.
exit :- halt.
e :- halt.
%---------------Locations----------------
location('bedroom').
location('corridor').
location('storage room').
location('kitchen').
location('garden').

connect('bedroom','corridor').
connect('corridor','bedroom').
connect('corridor','storage room').
connect('storage room','corridor').
connect('corridor','kitchen').
connect('kitchen','corridor').
connect('kitchen','garden').

exit('garden').

%---------------People-------------------
person(you).

%---------------Objects------------------
position(you,'bedroom').

object('window').
object('bed').
object('wardrobe').
object('table').
object('chimney').
object('picture of a garden').
object('picture of the sea').
object('picture of an apple').
object('bedroom door').
object('storage room door').
object('kitchen door').
object('bedroom key').
object('small lockpick').
object('wine barrel').
object('wooden crate').
object('suspicious wine barrel').
object('broken furniture').
object('salt box').
object('exit').
object('key with gear').
object('old clock').
object('dead rat').
object('exit key').
	
describes('window','Can see a countryside.').
describes('bed','The bed looks awfull.').
describes('wardrobe','Large wooden wardrobe. People store things there. There might be something in it.').
describes('table','Clean oak table. There are some drawers in it, but looks like they will not open.').
describes('chimney','Old chimney with a pile of ashes.').
describes('picture of a garden','You can see the same countryside as in the window. The sun shines brightly. Children play with dolls.').
describes('picture of the sea','You can see the storm in the sea with a ship strugling to float.').
describes('picture of an apple','This apple is good.').
describes('bedroom door','This is the door in the bedroom. Can be unlocked with a key.').
describes('storage room door','This is the door in the storage. The keyhole is broken, so the door is always opened.').
describes('bedroom key', 'This is the key to the bedroom door.').
describes('small lockpick','This can be used to open a simple small lock.').
describes('wine barrel','A big barrel full of expensive wine. Needs a hammer to open it. Doubt I can find it though.').
describes('wooden crate','A solid wooden crate. Smells of dirt.').
describes('suspicious wine barrel','Looks like there is no wine in it. The side could be firmly opened.').
describes('kitchen door','A big door to the kitchen. The lock looks simple.').
describes('broken furniture','All the furniture in the kitchen is completely broken').
describes('salt box','A box of salt, something could be in there').
describes('key with gear','A beautifull old key. Could be used to open something such old and beautifull.').
describes('old clock','An old beautifull clock, but it does not go. Maybe something is stuck in the gears. Could find a way to open it.').
describes('dead rat','Really stinks...').
describes('exit key','A key to the exit doors.').

%interractables
pickable('bedroom key').
pickable('small lockpick').
pickable('key with gear').
pickable('exit key').

openable('wardrobe').
openable('bedroom door').
openable('kitchen door').
openable('suspicious wine barrel').
openable('salt box').
openable('old clock').

container('wardrobe').
container('suspicious wine barrel').
container('salt box').
container('old clock').
contains('wardrobe','bedroom key').
contains('suspicious wine barrel','small lockpick').
contains('salt box','key with gear').
contains('old clock','exit key').

%doors
entrance('bedroom','bedroom door').
entrance('corridor','bedroom door').
entrance('corridor','storage room door').
entrance('storage room','storage room door').
entrance('corridor','kitchen door').
entrance('kitchen','kitchen door').
entrance('garden','exit').
entrance('kitchen','exit').

closed('bedroom door','bedroom key').
closed('kitchen door','small lockpick').
closed('old clock','key with gear').

%bedroom
positioned('window','bedroom').
%positioned('key with gear','bedroom').
positioned('bed','bedroom').
positioned('wardrobe','bedroom').
positioned('table','bedroom').
positioned('chimney','bedroom').
positioned('picture of a garden','bedroom').
positioned('bedroom door','bedroom').
positioned('old clock','bedroom').
%corridor
positioned('bedroom door','corridor').
positioned('storage room door','corridor').
positioned('kitchen door','corridor').
positioned('window','corridor').
positioned('picture of the sea','corridor').
positioned('picture of an apple','corridor').
%storage
positioned('storage room door','storage room').
positioned('wooden crate','storage room').
positioned('wine barrel','storage room').
positioned('wine barrel','storage room').
positioned('wine barrel','storage room').
positioned('suspicious wine barrel','storage room').
positioned('wine barrel','storage room').
positioned('wooden crate','storage room').
positioned('wooden crate','storage room').
positioned('dead rat','storage room').
%kitchen
positioned('kitchen door','kitchen').
positioned('exit','kitchen').
positioned('broken furniture','kitchen').
positioned('salt box','kitchen').
bag([]).

%--------------Actions----------------

whereAmI :- getPosition.
getPosition :- nl, write(' You are currently in the '), position(you,X), write(X).
listConnections :- write(' You can go to: '), nl, position(you,X), connect(X,Y), tab(2), write(Y), nl, fail.
listConnections :- nl.
listObjects :- write(' You can see:'), nl, position(you,X), positioned(Y,X), tab(2), write(Y), nl, fail. 
listObjects :- nl.

look :- getPosition, nl, listConnections, listObjects.
inspect(X) :- position(you, Pos), positioned(X,Pos), describes(X,Y),write(Y),nl,!.
inspect(X) :- write(' There is no such object in this location.'),nl,!.

goto(P) :- 
	position(you,X), 
	entrance(P,Y),
	entrance(X,Y),
	connect(X,P),
	closed(Y,K),
	write(Y),
	write(' Cannot go there because the doors are closed.'),nl,!.

goto(P) :- 
	position(you,X), 
	entrance(P,Y),
	entrance(X,Y),
	connect(X,P),
	exit(P),
	write(' You have escaped the building and quickly went home.'),
	nl,
	write(' The end.'),
	halt.

goto(P) :- 
	position(you,X), 
	entrance(P,Y),
	entrance(X,Y),
	connect(X,P),
	retract(position(you,X)),
	assert(position(you,P)),
	look,!.

goto(P) :- write(' You can not go there.'),nl,!.

writeList([Head|Tail]) :- tab(2), write(Head), nl, writeList(Tail).

inventory :- write(' Content of your bag: '),nl, bag(X), writeList(X), !.
inventory :- write(' This was the all content of your bag.').

pickUp(Y) :- 
	position(you,Pos),
	positioned(Y,Pos),
	pickable(Y),
	retract(positioned(Y,Pos)),
	bag(X), 
	append(X, [Y], Z), 
	retractall(bag(_)), 
	assert(bag(Z)), 
	write(' You have picked up '), 
	write(Y), 
	nl, !.
pickUp(Y) :- object(Y), write(' Object you are trying to pick up is not here'), !.
pickUp(Y) :- write(' There is no such object.'),nl, !.
drop(Y) :- 
	bag(X), 
	member(Y,X),
	delete(X,Y,Z), 
	retractall(bag(_)), 
	assert(bag(Z)), 
	position(you,Pos),
	assert(positioned(Y,Pos)),
	write(' You have dropped '), 
	write(Y), 
	nl.
drop(Y) :- write(' There is no such object in your inventory.'),nl, !.
use(Y) :- 
	bag(X),
	member(Y,X), 
	delete(X,Y,Z), 
	retractall(bag(_)), 
	assert(bag(Z)), 
	position(you,Pos),
	write(' You have used '), 
	write(Y), 
	nl.
use(Y) :- write(' There is no object in your inventory to use.'),nl,fail.

open(X):-
	openable(X),
	position(you,Pos),
	positioned(X,Pos),
	closed(X,K),
	not(contains(X,Y)),
	retract(closed(X,K)),
	use(K),
	write(' The '),
	write(X), 
	write(' is now opened.'),
	nl, !.

open(X):-
	openable(X),
	container(X),
	position(you,Pos),
	closed(X,K),
	contains(X,Y),
	use(K),
	assert(positioned(Y,Pos)),
	write(' There was a '),
	write(Y),
	write(' in the '),
	write(X),
	nl,!.

open(X):-
	openable(X),
	container(X),
	position(you,Pos),
	contains(X,Y),
	not(closed(X,K)),
	assert(positioned(Y,Pos)),
	write(' There was a '),
	write(Y),
	write(' in the '),
	write(X),
	nl,!.

open(X) :- write('Cannot open the '),write(X),nl,!.

%---------------Quest-------------------
quest(no,0).
day(1).
getTime :- nl, write(' It is the day '), day(D), write(D), write(' since the beginning of your adventure.').
passDay :- day(D), retract(day(D)),D1 is D + 1, assert(day(D1)).

%--------------Help---------------------
help :- 
	write(' Write "newGame" to start the game'),
	nl,
	write(' Write "look" to look around'),
	nl,
	write(' Write "inventory" to display content of your bag'),
	nl,
	write(' Write "goto(location)" to go to the location'),
	nl,
	write(' Write "pickUp(object)" to pick up object'),
	nl,
	write(' Write "drop(object)" to drop object'),
	nl,
	write(' Write "open(object)" to open an object'),
	nl,
	write(' Write "inspect(object)" to analyse the object').

