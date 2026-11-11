from typing import List


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        merges = []
        # 1. Split corpus into a list of individual characters
        characters = list(corpus)
        # 2. For each merge step:
        for _ in range(num_merges):
            if len(characters) < 2:
                break
            # a. Count frequency of all adjacent token pairs
            freq = dict()
            max_freq = 0
            for i in range(len(characters) - 1):
                pair = (characters[i], characters[i + 1])
                freq[pair] = freq.get(pair, 0) + 1
                max_freq = max(max_freq, freq[pair])
            # b. Find the most frequent pair (break ties lexicographically)
            candidates = sorted(key for key, val in freq.items() if val == max_freq)
            tokens = candidates[0]
            new_token = tokens[0] + tokens[1]
            # c. Merge all non-overlapping occurrences left to right
            new_characters = []
            i = 0
            while i < len(characters):
                if (
                    i < len(characters) - 1 and
                    (characters[i], characters[i + 1]) == tokens
                ):
                    new_characters.append(new_token)
                    i += 2
                else:
                    new_characters.append(characters[i])
                    i += 1
            characters = new_characters
            # d. Record the merge as [token_a, token_b]
            merges.append(tokens)
        # 3. Return the list of merges performed
        return merges
