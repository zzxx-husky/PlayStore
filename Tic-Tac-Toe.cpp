#include <cstdio>
#include <iostream>
#include <algorithm>
#include <unordered_map>
#include <cassert>
using namespace std;

struct pairhash {
public:
	template <typename T, typename U>
	size_t operator()(const pair<T, U> &x) const
	{
		return hash<T>()(x.first) ^ hash<U>()(x.second);
	}
};

const int NOTEND = 0;
const int OWIN = 1;
const int XWIN = 2;
const int DRAW = 3;
const int MAXITER = 50000000;
const double gamma = 0.9;
const double WINREWARD = 20;
const double LOSEREWARD = 0;
const double DUALREWARD = 10;
const int AVGN = 100;
const bool LEARN = true;
const bool REPORTAVG = false;
const double RANDCHOOSE = 0; // 0.05;
unordered_map<pair<int, int>, double, pairhash> Q;
unordered_map<pair<int, int>, int, pairhash> V;

/*
 * board: 19 bits.
 * the top bit: 0 for O, 1 for X
 * first 9 bits: 0 no filled, 1 filled
 * last 9 bits: 0 filled with O, 1 filled with X
 */
int game_status(int board) {
	static bool inited = false;
	static int wins[9] = { 0 };
	if (!inited) {
		int w[9] = {
			111000000, 111000, 111, // rows
			100100100, 10010010, 1001001, // columns
			100010001, 1010100, 111111111, // diagonals
		};
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				wins[i] = (wins[i] << 1) | (w[i] % 10);
				w[i] /= 10;
			}
		}
		inited = true;
	}
	for (int i = 0; i < 8; i++) {
		if (((board >> 9) & wins[i]) == wins[i]) {
			// printf("A: %d\n", i);
			int filled = board & wins[i];
			if (filled == wins[i] || filled == 0) {
				return filled == wins[i] ? XWIN : OWIN;
			}
		}
	}
	return ((board >> 9) & wins[8]) == wins[8] ? DRAW : NOTEND;
}

// if is player 1, flip O to X and X to O
int turn_board(int board, int player) {
	return player == 0 ? board : (board ^ ((1 << 9) - 1) ^ (1 << 18));
}

int select_action(int board, int player) {
	int cand[9] = { 0 };
	double prob[9] = { 0 };
	int cn = 0;

	board = turn_board(board, player);

	for (int i = 0; i < 9; i++) {
		if ((board & (1 << (i + 9))) == 0) { // not filled
			cand[cn] = i;
			cn++;
		}
	}
	if (cn == 0) {
		throw exception("No available actions");
	}
	if (rand() % 10000 / 10000. > RANDCHOOSE) {
		for (int i = 0; i < cn; i++) {
			prob[i] = exp(Q[{board, cand[i]}]);
			if (i != 0) {
				prob[i] += prob[i - 1];
			}
		}
		double choose = rand() % 10000 / 10000. * prob[cn - 1];
		for (int i = 0; i < cn - 1; i++) {
			if (choose <= prob[i])
				return cand[i];
		}
		return cand[cn - 1];
	}
	else {
		return cand[rand() % cn];
	}
}

int apply_action(int board, int action, int player) {
	// set position filled, set what's filled, switch player
	return (board | (1 << (action + 9)) | (player << action)) ^ (1 << 18);
}

double maxQ(int board) {
	double res = 0;
	for (int i = 0; i < 9; i++) {
		if ((board & (1 << (i + 9))) == 0) { // not filled
			res = max(res, Q[{board, i}]);
		}
	}
	return res;
}

class ShowBoard {
public:
	void init() {
		for (int i = 0; i < 3; i++) {
			boards[i].clear();
		}
	}
	void record(int board) {
		static const char* spaces = "    ";
		static const char* arrow = " -> ";
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				boards[i].push_back((board & (1 << (i * 3 + j + 9))) ?
					board & (1 << (i * 3 + j)) ? 'X' : 'O' : ' ');
			}
		}
		boards[0].append(spaces, spaces + 4);
		boards[1].append(arrow, arrow + 4);
		boards[2].append(spaces, spaces + 4);
	}
	void show(int status) {
		for (int i = 0; i < 3; i++) {
			printf("%s%s\n", boards[i].data(), i != 1 ? "" :
				status == DRAW ? "DRAW" :
				status == XWIN ? "X wins" :
				status == OWIN ? "O wins" : "NOTEND");
		}
		printf("\n");
	}
private:
	string boards[3];
};

void train() {
	ShowBoard boarder;
	int draw = 0, xwin = 0, owin = 0, total = 0;
	int records[AVGN], rn = 0;
	for (int iter = MAXITER; iter--;) {
		int board = 0, status, actn[2] = { 0 }, player = 1;
		pair<int, int> acts[2][9];

		// boarder.init();
		for (; (status = game_status(board)) == NOTEND;) {
			player ^= 1;
			int action = select_action(board, player);
			acts[player][actn[player]++] = { turn_board(board, player), action };
			++V[{turn_board(board, player), action}];
			board = apply_action(board, action, player);
			// boarder.record(board);
		}
		// boarder.show(status);

		status == DRAW ? ++draw : status == XWIN ? ++xwin : ++owin;
		total < AVGN ? ++total : records[rn] == DRAW ? --draw : records[rn] == XWIN ? --xwin : --owin;
		records[rn++] = status;
		rn %= AVGN;
		if (REPORTAVG || iter == 0) {
			printf("Dual ratio: %.2f\n", draw / (total + 0.));
			printf("X win ratio: %.2f\n", xwin / (total + 0.));
			printf("O win ratio: %.2f\n", owin / (total + 0.));
			printf("\n");
		}

		if (LEARN) {
			if (status != DRAW) {
				auto a = acts[player][actn[player] - 1];
				double alpha = 1. / (1 + V[a]);
				Q[a] = (1 - alpha) * Q[a] + alpha * WINREWARD;

				a = acts[player ^ 1][actn[player ^ 1] - 1];
				alpha = 1. / (1 + V[a]);
				Q[a] = (1 - alpha) * Q[a] + alpha * LOSEREWARD;
			}
			else {
				auto a = acts[0][actn[0] - 1];
				double alpha = 1. / (1 + V[a]);
				Q[a] = (1 - alpha) * Q[a] + alpha * DUALREWARD;

				a = acts[1][actn[1] - 1];
				alpha = 1. / (1 + V[a]);
				Q[a] = (1 - alpha) * Q[a] + alpha * DUALREWARD;
			}
			for (int v = 0; v < 2; v++) {
				for (int i = actn[v] - 2; i >= 0; i--) {
					auto a = acts[v][i];
					double alpha = 1. / (1 + V[a]);
					Q[a] = (1 - alpha) * Q[a] + alpha * gamma * maxQ(acts[v][i + 1].first);
					assert(!std::isinf(Q[a]));
				}
			}
		}
	}
}

int main() {
	freopen("a.out", "w", stdout);
	train();
}