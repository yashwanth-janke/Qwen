def findMaximumBandwidths(n, k, m):
    def get_min_sum_for_side(length, peak_val):
        if peak_val <= 1:
            return length
        
        decreasing_count = min(length, peak_val - 1)
        
        first = peak_val - 1
        last = peak_val - decreasing_count
        sum_decreasing = decreasing_count * (first + last) // 2
        
        ones_count = length - decreasing_count
        
        return sum_decreasing + ones_count
    def is_possible(x):
        required = x
        
        if k > 1:
            required += get_min_sum_for_side(k - 1, x)
        
        if k < n:
            required += get_min_sum_for_side(n - k, x)
            
        return required <= m

    low, high = 1, m
    ans = 1
    
    while low <= high:
        mid = (low + high) // 2
        if mid == 0:
            low = 1
            continue
            
        if is_possible(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
            
    return ans

print(findMaximumBandwidths(6,2,11))  # Output: 2