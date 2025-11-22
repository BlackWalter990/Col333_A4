#include <iostream>
#include <vector>
#include <algorithm>

typedef long long ll;

using namespace std;
ll solve(vector<int> stones) {
    int n = stones.size();
    if (n <= 1) {
        return 0;
    }
    std::vector<ll> dp(n, 0);
    dp[0] = 0;
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            ll current_score = dp[j] + stones[i] * (ll)(i - j);
            if (current_score > dp[i]) {
                dp[i] = current_score;
            }
        }
    }
    return dp[n - 1];
}

int main() {
    vector<int> stones = {3, 7, 2, 10, 5, 12, 8, 10, 1};
    ll result = solve(stones);
    std::cout << result << std::endl;
    return 0;
}
