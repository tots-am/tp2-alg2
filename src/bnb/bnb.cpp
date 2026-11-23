// bnb.cpp — Branch-and-Bound exato para o TSP simétrico (C++11).
//
// Entrada (stdin):  n
//                   matriz n x n de inteiros (distâncias EUC_2D já arredondadas)
//                   custo do tour inicial (upper bound do warm-start)
//                   tour inicial (n inteiros, 0-based)
// Argumentos:       --bound naive|mst|hk   (cota inferior)
//                   --time-limit S         (segundos; ao estourar, para e reporta)
// Saída (stdout):   uma linha JSON com custo, tour, contadores da árvore e tempos.
//
// Estratégia de busca: DFS (profundidade), filhos ordenados por distância ao
// último vértice (vizinho mais próximo primeiro). DFS mantém memória O(n)
// (apenas o caminho atual), ao contrário de best-first cuja fila cresce
// exponencialmente. O estado é incremental: um vetor de caminho e um vetor de
// visitados, sem cópias de matriz por nó.
//
// Cotas inferiores para um caminho parcial v0 -> ... -> vk com U = não visitados:
//   naive: custo(caminho) + ( m1[vk] + m1[v0] + sum_{u in U}(m1[u]+m2[u]) ) / 2
//          m1/m2 = 1a e 2a menor aresta incidente (pré-computadas, O(1) por nó
//          com soma incremental). É a cota "clássica" de livro-texto.
//   mst:   custo(caminho) + MST(U) + min d(vk,U) + min d(U,v0)
//          (o restante do tour é um caminho hamiltoniano vk -> U -> v0;
//           removidas as duas arestas das pontas sobra uma árvore geradora de U)
//   hk:    a mesma cota mst sobre custos d'(i,j) = d(i,j) + pi_i + pi_j, com
//          pi obtido UMA vez na raiz por otimização de subgradiente (Held-Karp).
//          Válida para qualquer pi: LB = LB'(resto) - pi_vk - pi_v0 - 2 sum_U pi.
//          O passo do subgradiente NÃO depende do UB, para que o warm-start
//          influencie a busca apenas via poda (isola o efeito experimental).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <sys/resource.h>

typedef long long ll;
typedef std::chrono::steady_clock Clock;

enum BoundType { NAIVE, MST, HK };

static int n;
static std::vector<int> D;          // matriz de distâncias (row-major)
static std::vector<double> Dp;      // d'(i,j) = d + pi_i + pi_j (só para hk)
static std::vector<double> pi_;     // penalidades de Held-Karp
static std::vector<int> order_;     // order_[v*n + k] = k-ésimo vizinho mais próximo de v
static std::vector<int> m1, m2;     // menores arestas incidentes (cota naive)

static BoundType bound_type = MST;
static double time_limit = 1800.0;

// Estado da busca
static std::vector<int> path_;
static std::vector<char> visited;
static std::vector<int> best_tour;
static ll UB;
static ll naive_sumU;               // soma incremental de (m1+m2) sobre U

// Estatísticas
static ll nodes_expanded = 0;       // nós cujos filhos foram gerados
static ll nodes_pruned = 0;         // filhos descartados por cota >= UB
static ll leaves = 0;               // tours completos avaliados
static ll bound_calls = 0;
static ll ub_updates = 0;
static int max_depth = 0;
static double t_best = 0.0;         // instante em que o melhor tour foi achado
static double t_bound = 0.0;        // tempo acumulado na função de cota (amostrado)
static bool timed_out = false;
static Clock::time_point t0;

// buffers reutilizados (evitam alocação por nó)
static std::vector<int> ubuf;
static std::vector<double> keyd;
static std::vector<char> inT;

static inline int d(int i, int j) { return D[(size_t)i * n + j]; }

static double elapsed() {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// --------------------------------------------------------------------------
// Prim denso O(m^2) sobre um conjunto de vértices, com custos d ou d'.
// --------------------------------------------------------------------------
template <typename W>
static double mst_cost(const int* V, int m, W w) {
    if (m <= 1) return 0.0;
    for (int i = 0; i < m; ++i) { keyd[i] = std::numeric_limits<double>::infinity(); inT[i] = 0; }
    keyd[0] = 0.0;
    double total = 0.0;
    for (int it = 0; it < m; ++it) {
        int bi = -1; double bk = std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i)
            if (!inT[i] && keyd[i] < bk) { bk = keyd[i]; bi = i; }
        inT[bi] = 1; total += bk;
        int vb = V[bi];
        for (int i = 0; i < m; ++i)
            if (!inT[i]) {
                double c = w(vb, V[i]);
                if (c < keyd[i]) keyd[i] = c;
            }
    }
    return total;
}

// --------------------------------------------------------------------------
// Cota inferior do subproblema (caminho fixo path_[0..depth-1], custo 'cost').
// --------------------------------------------------------------------------
static double lower_bound(int last, int depth, ll cost) {
    const int v0 = path_[0];
    if (depth == n) return (double)(cost + d(last, v0));

    if (bound_type == NAIVE) {
        return cost + 0.5 * (double)(m1[last] + m1[v0] + naive_sumU);
    }

    int m = 0;
    for (int v = 0; v < n; ++v) if (!visited[v]) ubuf[m++] = v;

    if (bound_type == MST) {
        int a = std::numeric_limits<int>::max(), b = a;
        for (int i = 0; i < m; ++i) {
            a = std::min(a, d(last, ubuf[i]));
            b = std::min(b, d(ubuf[i], v0));
        }
        double t = mst_cost(ubuf.data(), m, [](int i, int j) { return (double)d(i, j); });
        return cost + t + a + b;
    }

    // HK: custos modificados
    double a = std::numeric_limits<double>::infinity(), b = a, sp = 0.0;
    for (int i = 0; i < m; ++i) {
        int u = ubuf[i];
        a = std::min(a, Dp[(size_t)last * n + u]);
        b = std::min(b, Dp[(size_t)u * n + v0]);
        sp += pi_[u];
    }
    double t = mst_cost(ubuf.data(), m, [](int i, int j) { return Dp[(size_t)i * n + j]; });
    return cost + t + a + b - pi_[last] - pi_[v0] - 2.0 * sp;
}

// --------------------------------------------------------------------------
// Held-Karp na raiz: 1-tree (MST em {1..n-1} + 2 menores arestas de 0).
// Passo geométrico independente do UB. Retorna a melhor cota encontrada.
// --------------------------------------------------------------------------
static double held_karp_root(int iters) {
    pi_.assign(n, 0.0);
    std::vector<double> best_pi(n, 0.0), w(n * (size_t)n);
    std::vector<int> deg(n), parent(n);
    std::vector<double> key(n);
    std::vector<char> in(n);
    double best = -1e300;

    // passo inicial proporcional ao comprimento médio de aresta
    double mean = 0; for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) mean += d(i, j);
    mean /= (double)n * (n - 1);
    double step = 0.1 * mean;
    int since_improve = 0;

    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
            w[(size_t)i * n + j] = d(i, j) + pi_[i] + pi_[j];
        // Prim em {1..n-1}
        std::fill(deg.begin(), deg.end(), 0);
        std::fill(in.begin(), in.end(), 0);
        std::fill(key.begin(), key.end(), 1e300);
        key[1] = 0; parent[1] = -1;
        double L = 0;
        for (int k = 1; k < n; ++k) {
            int bi = -1; double bk = 1e300;
            for (int v = 1; v < n; ++v) if (!in[v] && key[v] < bk) { bk = key[v]; bi = v; }
            in[bi] = 1; L += bk;
            if (parent[bi] >= 0) { deg[bi]++; deg[parent[bi]]++; }
            for (int v = 1; v < n; ++v)
                if (!in[v] && w[(size_t)bi * n + v] < key[v]) { key[v] = w[(size_t)bi * n + v]; parent[v] = bi; }
        }
        // duas menores arestas a partir de 0
        int a = -1, b = -1;
        for (int v = 1; v < n; ++v) {
            if (a < 0 || w[v] < w[a]) { b = a; a = v; }
            else if (b < 0 || w[v] < w[b]) b = v;
        }
        L += w[a] + w[b]; deg[0] = 2; deg[a]++; deg[b]++;
        double sp = 0; for (int i = 0; i < n; ++i) sp += pi_[i];
        L -= 2.0 * sp;
        if (L > best + 1e-9) { best = L; best_pi = pi_; since_improve = 0; }
        else if (++since_improve >= std::max(5, n / 4)) { step *= 0.5; since_improve = 0; }
        double norm = 0; for (int i = 0; i < n; ++i) norm += (double)(deg[i] - 2) * (deg[i] - 2);
        if (norm == 0) break;            // 1-tree é um tour: ótimo provado
        if (step < 1e-6) break;
        for (int i = 0; i < n; ++i) pi_[i] += step * (deg[i] - 2);
    }
    pi_ = best_pi;
    Dp.resize((size_t)n * n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
        Dp[(size_t)i * n + j] = d(i, j) + pi_[i] + pi_[j];
    return best;
}

// --------------------------------------------------------------------------
// DFS
// --------------------------------------------------------------------------
static inline bool prune_test(double lb) {
    // custos inteiros: se ceil(lb) >= UB, nenhum tour estritamente melhor existe
    return std::ceil(lb - 1e-7) >= (double)UB;
}

static void dfs(int last, int depth, ll cost) {
    if (timed_out) return;
    ++nodes_expanded;
    if (depth > max_depth) max_depth = depth;

    const int* nb = &order_[(size_t)last * n];
    for (int k = 0; k < n; ++k) {
        int u = nb[k];
        if (visited[u]) continue;
        ll nc = cost + d(last, u);

        if (depth + 1 == n) {            // folha: fecha o ciclo
            ++leaves;
            ll total = nc + d(u, path_[0]);
            if (total < UB) {
                UB = total; ++ub_updates; t_best = elapsed();
                path_[depth] = u;
                best_tour.assign(path_.begin(), path_.end());
            }
            continue;
        }
        if (nc >= UB) { ++nodes_pruned; continue; }

        visited[u] = 1; path_[depth] = u;
        naive_sumU -= (m1[u] + m2[u]);
        double lb;
        if ((++bound_calls & 255) == 0) {            // amostra 1/256 para medir custo da cota
            Clock::time_point s = Clock::now();
            lb = lower_bound(u, depth + 1, nc);
            t_bound += 256.0 * std::chrono::duration<double>(Clock::now() - s).count();
            if (elapsed() > time_limit) timed_out = true;
        } else {
            lb = lower_bound(u, depth + 1, nc);
        }
        if (prune_test(lb)) ++nodes_pruned;
        else dfs(u, depth + 1, nc);
        naive_sumU += (m1[u] + m2[u]);
        visited[u] = 0;
        if (timed_out) return;
    }
}

int main(int argc, char** argv) {
    std::string bstr = "mst";
    int hk_iters = 2000;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--bound") && i + 1 < argc) bstr = argv[++i];
        else if (!strcmp(argv[i], "--time-limit") && i + 1 < argc) time_limit = atof(argv[++i]);
        else if (!strcmp(argv[i], "--hk-iters") && i + 1 < argc) hk_iters = atoi(argv[++i]);
    }
    if (bstr == "naive") bound_type = NAIVE;
    else if (bstr == "hk") bound_type = HK;
    else bound_type = MST;

    if (scanf("%d", &n) != 1 || n < 3) { fprintf(stderr, "entrada inválida\n"); return 1; }
    D.resize((size_t)n * n);
    for (size_t i = 0; i < D.size(); ++i) if (scanf("%d", &D[i]) != 1) return 1;
    ll ub0; if (scanf("%lld", &ub0) != 1) return 1;
    std::vector<int> tour0(n);
    for (int i = 0; i < n; ++i) if (scanf("%d", &tour0[i]) != 1) return 1;

    t0 = Clock::now();

    // vizinhos ordenados por distância
    order_.resize((size_t)n * n);
    for (int v = 0; v < n; ++v) {
        int* o = &order_[(size_t)v * n];
        for (int k = 0; k < n; ++k) o[k] = k;
        std::sort(o, o + n, [v](int a, int b) { return d(v, a) < d(v, b) || (d(v, a) == d(v, b) && a < b); });
    }
    m1.assign(n, 0); m2.assign(n, 0);
    for (int v = 0; v < n; ++v) {   // order_[v*n+0] é o próprio v (distância 0)
        int c = 0;
        for (int k = 0; k < n && c < 2; ++k) {
            int u = order_[(size_t)v * n + k];
            if (u == v) continue;
            if (c == 0) m1[v] = d(v, u); else m2[v] = d(v, u);
            ++c;
        }
    }

    double t_hk = 0, root_lb_hk = 0;
    if (bound_type == HK) {
        Clock::time_point s = Clock::now();
        root_lb_hk = held_karp_root(hk_iters);
        t_hk = std::chrono::duration<double>(Clock::now() - s).count();
    }

    ubuf.resize(n); keyd.resize(n); inT.resize(n);
    path_.assign(n, -1); visited.assign(n, 0);
    UB = ub0; best_tour = tour0;
    path_[0] = 0; visited[0] = 1;
    naive_sumU = 0;
    for (int v = 1; v < n; ++v) naive_sumU += m1[v] + m2[v];

    double root_lb = lower_bound(0, 1, 0);
    if (!prune_test(root_lb)) dfs(0, 1, 0);
    double total = elapsed();

    struct rusage ru; getrusage(RUSAGE_SELF, &ru);

    printf("{\"cost\":%lld,\"initial_ub\":%lld,\"optimal\":%s,\"timed_out\":%s,"
           "\"nodes_expanded\":%lld,\"nodes_pruned\":%lld,\"leaves\":%lld,\"bound_calls\":%lld,"
           "\"ub_updates\":%lld,\"max_depth\":%d,\"root_lb\":%.4f,\"root_lb_hk\":%.4f,"
           "\"time_total\":%.6f,\"time_hk_root\":%.6f,\"time_bound_est\":%.6f,\"time_to_best\":%.6f,"
           "\"maxrss_kb\":%ld,\"tour\":[",
           UB, ub0, timed_out ? "false" : "true", timed_out ? "true" : "false",
           nodes_expanded, nodes_pruned, leaves, bound_calls, ub_updates, max_depth,
           root_lb, root_lb_hk, total, t_hk, t_bound, t_best, ru.ru_maxrss);
    for (int i = 0; i < n; ++i) printf(i ? ",%d" : "%d", best_tour[i]);
    printf("]}\n");
    return 0;
}
