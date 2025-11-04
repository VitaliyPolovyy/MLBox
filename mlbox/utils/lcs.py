"""
Longest Common Subsequence (LCS) and text matching algorithms.

This module provides utilities for finding common subsequences between texts,
including suffix automaton implementation for efficient substring matching.
"""

from dataclasses import dataclass
from typing import List
import re


@dataclass
class Match:
    text: str
    len_a: int  # довжина у ref
    len_b: int  # довжина у label
    start_a: int
    start_b: int
    
    


def apply_ignorable_symbols(matches: List[Match], ignorable_symbols: str, text_a: str, text_b: str) -> List[Match]:
    """
    Extend matches to include adjacent ignorable symbols (by 1 character).
    Checks the symbol before start and after end, and extends if it's ignorable.
    
    Args:
        matches: List of Match objects
        ignorable_symbols: String of symbols to ignore (e.g., ",.()!:;- ")
        ref: Reference text (to check boundaries)
        label: Label text (to check boundaries)
    
    Returns:
        List of Match objects with extended boundaries
    """
    extended_matches = []
    
    for match in matches:
        start_a = match.start_a
        end_a = match.start_a + match.len_a
        start_b = match.start_b
        end_b = match.start_b + match.len_b
        symbol = ''
        text = match.text
        
        # Check and extend by 1 before start in ref
        if start_a > 0 and text_a[start_a - 1] in ignorable_symbols:
            symbol = text_a[start_a - 1]
            if start_b > 0 and text_b[start_b - 1] == symbol: 
                start_b -= 1
                start_a -= 1
                text = symbol + text
                
        symbol = ''
        # Check and extend by 1 after end in ref
        if end_a < len(text_a) and text_a[end_a] in ignorable_symbols:
            symbol = text_a[end_a]
            if end_b < len(text_b) and text_b[end_b] == symbol: 
                end_b += 1
                end_a += 1
                text = text + symbol


        # Create extended match
        extended_matches.append(Match(
            text=text,
            len_a=end_a - start_a,
            len_b=end_b - start_b,
            start_a=start_a,
            start_b=start_b
        ))
    
    return extended_matches


def filter_maximal_matches(matches: List[Match]) -> List[Match]:
    """
    Filter matches to keep only maximal ones - remove overlapping matches.
    For overlapping matches, keeps the longer one.
    """
    if not matches:
        return matches
    
    # Sort by length descending to prioritize longer matches
    sorted_matches = sorted(matches, key=lambda m: m.len_a, reverse=True)
    maximal_matches = []
    
    for current_match in sorted_matches:
        is_maximal = True
        
        # Check if this match overlaps with any already accepted maximal match
        for maximal_match in maximal_matches:
            # Check for overlap in reference text
            current_start_a = current_match.start_a
            current_end_a = current_match.start_a + current_match.len_a
            maximal_start_a = maximal_match.start_a
            maximal_end_a = maximal_match.start_a + maximal_match.len_a
            
            # Check for overlap in label text
            current_start_b = current_match.start_b
            current_end_b = current_match.start_b + current_match.len_b
            maximal_start_b = maximal_match.start_b
            maximal_end_b = maximal_match.start_b + maximal_match.len_b
            
            # Check if there's any overlap in either text
            overlap_a = not (current_end_a <= maximal_start_a or maximal_end_a <= current_start_a)
            overlap_b = not (current_end_b <= maximal_start_b or maximal_end_b <= current_start_b)
            
            if overlap_a or overlap_b:
                is_maximal = False
                break
        
        if is_maximal:
            maximal_matches.append(current_match)
    
    # Sort back by original order (by start positions)
    return sorted(maximal_matches, key=lambda m: (m.start_a, m.start_b))


def all_common_substrings_by_words(
    text_a: str, 
    text_b: str, 
    min_length_words=2, 
    maximal_only=False,
    ignorable_symbols: str = None
) -> List[Match]:
    """
    Повертає всі спільні підрядки (послідовності слів) довжиною >= min_length_words
    з позиціями у symbovih у ref та label.
    
    Args:
        ref: Reference text (etalon)
        label: Label text (actual)
        min_length_words: Minimum number of words for a match
        maximal_only: If True, filter to keep only maximal matches
        ignorable_symbols: String of symbols to ignore at match boundaries (e.g., ",.()!:;- ")
    """
    def tokenize_with_positions(text):
        words = []
        positions = []
        # Custom pattern: letters, digits, and domain-specific symbols
        for m in re.finditer(r'[\w°%]+', text, flags=re.UNICODE):
        #for m in re.finditer(r'\w+', text, flags=re.UNICODE):
            words.append(m.group(0))
            positions.append(m.start())
        return words, positions

    ref_words, ref_pos = tokenize_with_positions(text_a)
    label_words, label_pos = tokenize_with_positions(text_b)

    class WordSuffixAutomaton:
        def __init__(self):
            self.next = [{}]
            self.link = [-1]
            self.len = [0]
            self.last = 0

        def extend(self, token):
            p = self.last
            cur = len(self.next)
            self.next.append({})
            self.len.append(self.len[p] + 1)
            self.link.append(0)
            while p >= 0 and token not in self.next[p]:
                self.next[p][token] = cur
                p = self.link[p]
            if p == -1:
                self.link[cur] = 0
            else:
                q = self.next[p][token]
                if self.len[p] + 1 == self.len[q]:
                    self.link[cur] = q
                else:
                    clone = len(self.next)
                    self.next.append(self.next[q].copy())
                    self.len.append(self.len[p] + 1)
                    self.link.append(self.link[q])
                    while p >= 0 and self.next[p][token] == q:
                        self.next[p][token] = clone
                        p = self.link[p]
                    self.link[q] = self.link[cur] = clone
            self.last = cur

    sa = WordSuffixAutomaton()
    for w in ref_words:
        sa.extend(w)

    res = []
    n = len(label_words)
    v = 0
    l = 0
    for i in range(n):
        while v and label_words[i] not in sa.next[v]:
            v = sa.link[v]
            l = sa.len[v]
        if label_words[i] in sa.next[v]:
            v = sa.next[v][label_words[i]]
            l += 1
        else:
            v = 0
            l = 0
        if l >= min_length_words:
            pos_ref_word = -1
            vv = v
            while vv:
                if sa.len[sa.link[vv]] < min_length_words:
                    pos_ref_word = sa.len[vv] - l
                    break
                vv = sa.link[vv]
            if pos_ref_word == -1:
                pos_ref_word = i - l + 1
            # --- Перевірка коректності індексів ---
            if (
                0 <= pos_ref_word < len(ref_words) and
                0 <= pos_ref_word + l - 1 < len(ref_words) and
                0 <= i - l + 1 < len(label_words) and
                0 <= i < len(label_words)
            ):
                start_a = ref_pos[pos_ref_word]
                end_a = ref_pos[pos_ref_word + l - 1] + len(ref_words[pos_ref_word + l - 1])
                start_b = label_pos[i - l + 1]
                end_b = label_pos[i] + len(label_words[i])
                text = text_a[start_a:end_a]
                res.append(Match(
                    text=text,
                    len_a=end_a - start_a,
                    len_b=end_b - start_b,
                    start_a=start_a,
                    start_b=start_b
                ))
    
    # Apply maximal filtering if requested
    if maximal_only:
        res = filter_maximal_matches(res)
    

    result = filter_maximal_matches(res)

    if ignorable_symbols:
        result = apply_ignorable_symbols(result, ignorable_symbols, text_a, text_b)

    
    return result


def highlight_matches_by_words_html(text: str, matches: List[Match], use_start_a: bool = False) -> str:
    """
    Highlight matched text at WORD level (not character level).
    Entire words are either green (fully matched) or red (partially/fully unmatched).
    
    Args:
        text: The text to highlight
        matches: List of Match objects containing position information
        use_start_a: If True, use start_a and len_a from matches, otherwise use start_b and len_b
        
    Returns:
        HTML string with word-level highlighted text
    """
    import html
    
    if not matches:
        # No matches - entire text is unmatched (red)
        escaped_text = html.escape(text)
        return f'<span style="background-color: #ffcccc;">{escaped_text}</span>'
    
    result = ""
    start_pos = 0

    for match in matches:
        # Get the correct match position based on use_start_a parameter
        match_start = match.start_a if use_start_a else match.start_b
        match_len = match.len_a if use_start_a else match.len_b
        
        # Add unmatched text (red background)
        if start_pos < match_start:
            unmatched = html.escape(text[start_pos:match_start])
            result += f'<span style="background-color: #ffcccc;">{unmatched}</span>'
        
        # Add matched text (green background)
        matched = html.escape(text[match_start:match_start + match_len])
        result += f'<span style="background-color: #ccffcc;">{matched}</span>'
        start_pos = match_start + match_len

    # Add any remaining unmatched text (red background)
    if start_pos < len(text):
        remaining = html.escape(text[start_pos:])
        result += f'<span style="background-color: #ffcccc;">{remaining}</span>'
    
    return result


if __name__ == "__main__":
    ref =   "The quick brown fox jumps !over the! lazy dog"
    label = "The quick brown fox jumps over the lazy dog"
    print(all_common_substrings_by_words(ref, label, min_length_words=2, maximal_only=True))
