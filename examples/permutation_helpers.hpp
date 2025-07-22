inline void sort_perm(int *arr, int *perm, int len, bool rev = false) {
    if (rev == false) {
        std::stable_sort(perm + 0, perm + len, [&](const int &a, const int &b) {
            return (arr[a] < arr[b]);
        });
    } else {
        std::stable_sort(perm + 0, perm + len, [&](const int &a, const int &b) {
            return (arr[a] > arr[b]);
        });
    }
}

template <typename IT>
std::vector<IT> compute_sort_permutation(const std::vector<IT> &rows,
                                         const std::vector<IT> &cols) {
    std::vector<IT> perm(rows.size());
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
                     [&](std::size_t i, std::size_t j) {
                         if (rows[i] != rows[j])
                             return rows[i] < rows[j];
                         if (cols[i] != cols[j])
                             return cols[i] < cols[j];
                         return false;
                     });
    return perm;
}

template <typename IT, typename VT>
std::vector<VT> apply_permutation(std::vector<IT> &perm,
                                  std::vector<VT> &original) {
    std::vector<VT> sorted;
    sorted.reserve(original.size());
    std::transform(perm.begin(), perm.end(), std::back_inserter(sorted),
                   [&](auto i) { return original[i]; });
    original = std::vector<VT>();
    return sorted;
}