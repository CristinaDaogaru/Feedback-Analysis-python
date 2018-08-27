using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;

namespace FeedBackPlatformWeb.Models
{
    public class Category
    {
        [Key]
        public int Id { get; set; }
        [MaxLength(20)]
        public string Name { get; set; }

        public ICollection<Survey> Surveys { get; set; }
    }
}